"""Ollama + Z-Image Turbo App Studio - serveur local (bibliothèque standard uniquement).

Le serveur écoute uniquement sur 127.0.0.1. Il :
  - sert l'interface (dossier static/),
  - relaie le chat vers Ollama (streaming),
  - génère les images avec stable-diffusion.cpp (sd-cli, modèle Z-Image Turbo) à la demande,
  - sauvegarde / prévisualise les projets générés (dossier projects/).
"""
import base64
import html
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
PROJECTS_DIR = ROOT / "projects"

HOST = "127.0.0.1"
PORT = int(os.environ.get("STUDIO_PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
# Modèle Ollama qui enrichit les prompts avant la génération d'image (facultatif).
IMPROVE_MODEL = os.environ.get("IMPROVE_MODEL", "gnokit/improve-prompt")
IMPROVE_TIMEOUT = 300
IMPROVED_MAX = 800

# Lancement des programmes : chemins fixes issus de la configuration, jamais de la requête.
OLLAMA_EXE = os.environ.get("OLLAMA_EXE") or shutil.which("ollama") or str(
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
SD_DIR = Path(os.environ.get("SD_CPP_DIR", str(ROOT / "sd-cpp")))
SD_CLI = SD_DIR / "sd-cli.exe"
# Z-Image Turbo (GGUF quantifié) : modèle de diffusion + encodeur de texte Qwen3 4B + VAE FLUX.
SD_DIFFUSION = SD_DIR / "models" / "z_image_turbo-Q4_K.gguf"
SD_LLM = SD_DIR / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
SD_VAE = SD_DIR / "models" / "ae.safetensors"
SD_STEPS = 8  # Z-Image Turbo est distillé : 8 étapes, cfg 1.0
LOG_DIR = ROOT / "logs"

MAX_BODY = 20 * 1024 * 1024
CHAT_TIMEOUT = 600
IMAGE_TIMEOUT = 3600  # génération sur CPU : plusieurs minutes par image
LOCK_WAIT = 180  # attente max du verrou (rechargement d'Ollama ou image précédente en cours)
IMAGE_IDLE_RELOAD = 20  # secondes sans nouvelle image avant de recharger Ollama (laisse enchaîner un lot)
PING_TIMEOUT = 3

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_\-./]{0,119}$")
ALLOWED_EXT = {".html", ".css", ".js", ".json", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".md", ".txt"}
# Multiples de 16, tailles modestes : la génération se fait sur CPU.
ASPECTS = {"768*576", "1024*576", "768*768", "576*768", "576*1024"}

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("studio")


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- helpers
def http_json(url, payload=None, timeout=PING_TIMEOUT):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL de config locale)
        return json.loads(resp.read().decode("utf-8"))


def ollama_status():
    try:
        tags = http_json(OLLAMA_URL + "/api/tags")
        try:
            loaded = [m["name"] for m in http_json(OLLAMA_URL + "/api/ps").get("models", [])]
        except Exception:
            loaded = []
        return {"online": True, "models": [m["name"] for m in tags.get("models", [])], "loaded": loaded}
    except Exception:
        return {"online": False, "models": [], "loaded": []}


def installed():
    sd_files = (SD_CLI, SD_DIFFUSION, SD_LLM, SD_VAE)
    return {"ollama": Path(OLLAMA_EXE).is_file(), "images": all(p.is_file() for p in sd_files)}


# -- Détection du GPU. Deux niveaux : les périphériques que le sd-cli installé sait utiliser (--list-devices) et le
# matériel présent sur la machine (Windows, nvidia-smi), pour signaler qu'une version GPU de sd-cli est à installer.
SD_DEVICE = os.environ.get("SD_DEVICE", "auto")  # auto | cpu | nom d'un périphérique (ex. CUDA0, Vulkan0)
_DEVICE_RE = re.compile(r"[A-Za-z0-9_.:-]{1,40}")
_VIRTUAL_GPU_RE = re.compile(r"basic|remote|virtual|indirect|parsec|meta", re.I)
_gpu = {"devices": [], "hardware": [], "scanned": False, "scanning": False}
_gpu_lock = threading.Lock()


def _run_quiet(cmd, timeout, cwd=None):
    """Exécute une commande sans fenêtre et renvoie sa sortie standard ('' en cas d'échec)."""
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,  # noqa: S603
                       stdin=subprocess.DEVNULL, cwd=cwd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout if r.returncode == 0 else ""


def _scan_gpu():
    devices, hardware = [], []
    if SD_CLI.is_file():
        try:
            for line in _run_quiet([str(SD_CLI), "--list-devices"], 30, str(SD_DIR)).splitlines():
                name, _, desc = line.partition("\t")
                name = name.strip()
                if _DEVICE_RE.fullmatch(name) and not name.upper().startswith("CPU"):
                    devices.append({"name": name, "description": desc.strip()[:120]})
        except Exception:
            log.warning("Liste des périphériques sd-cli indisponible", exc_info=True)
    if sys.platform == "win32":
        try:
            out = _run_quiet(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              "(Get-CimInstance Win32_VideoController).Name"], 45)
            hardware = [{"name": n.strip()[:120], "vram_mb": None} for n in out.splitlines()
                        if n.strip() and not _VIRTUAL_GPU_RE.search(n)]
            if any("nvidia" in h["name"].lower() for h in hardware):
                smi = _run_quiet(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], 15)
                for line in smi.splitlines():
                    name, _, mem = line.rpartition(",")
                    for h in hardware:
                        if h["name"] == name.strip() and mem.strip().isdigit():
                            h["vram_mb"] = int(mem.strip())
        except Exception:
            log.warning("Détection du matériel graphique impossible", exc_info=True)
    with _gpu_lock:
        _gpu.update(devices=devices, hardware=hardware, scanned=True, scanning=False)
    log.info("GPU détecté : périphériques sd-cli=%s, matériel=%s", [d["name"] for d in devices],
             [h["name"] for h in hardware])


def start_gpu_scan():
    """Lance (une seule fois à la fois) la détection en arrière-plan : elle peut durer plusieurs secondes."""
    with _gpu_lock:
        if _gpu["scanning"]:
            return False
        _gpu["scanning"] = True
    threading.Thread(target=_scan_gpu, daemon=True).start()
    return True


def gpu_status():
    with _gpu_lock:
        snap = {k: (list(v) if isinstance(v, list) else v) for k, v in _gpu.items()}
    snap["default"] = SD_DEVICE
    return snap


def resolve_device(requested):
    """Périphérique de calcul à utiliser : 'cpu' ou le nom d'un périphérique détecté (liste d'autorisation)."""
    choice = SD_DEVICE if requested is None else requested
    if not isinstance(choice, str):
        raise ApiError(400, "Périphérique de calcul invalide.")
    names = [d["name"] for d in gpu_status()["devices"]]
    if choice == "cpu":
        return "cpu"
    if choice == "auto":
        return names[0] if names else "cpu"
    if choice in names:
        return choice
    raise ApiError(400, "Périphérique de calcul invalide.")


# Un seul gros modèle en mémoire à la fois : chat Ollama et génération d'image s'excluent mutuellement.
_mem_lock = threading.Lock()


def image_status():
    busy = _mem_lock.locked() and _generating.is_set()
    progress = None
    if busy and _progress:
        progress = {k: v for k, v in _progress.items() if k != "started"}
        progress["elapsed"] = int(time.monotonic() - _progress["started"])
    return {"busy": busy, "progress": progress}


_generating = threading.Event()  # une image est en cours de calcul (sd-cli lancé)


def reload_ollama_model(name):
    """Recharge un modèle Ollama en mémoire (requête sans prompt) une fois les images terminées."""
    try:
        http_json(OLLAMA_URL + "/api/generate", {"model": name, "keep_alive": "5m"}, 900)
        log.info("Ollama rechargé")
    except Exception:
        log.warning("Rechargement Ollama impossible", exc_info=True)


_idle_timer = None


def _cancel_idle_release():
    global _idle_timer
    timer, _idle_timer = _idle_timer, None
    if timer is not None:
        timer.cancel()


def _finish_images(model):
    """Fin d'un lot d'images : recharge le modèle Ollama."""
    if not _mem_lock.acquire(blocking=False):
        return  # une autre opération a pris la main : elle gère elle-même les modèles
    try:
        if model:
            reload_ollama_model(model)
    except Exception:
        log.error("Fin de lot d'images en erreur", exc_info=True)
    finally:
        _mem_lock.release()


def _arm_idle_release(model):
    global _idle_timer
    _cancel_idle_release()
    timer = threading.Timer(IMAGE_IDLE_RELOAD, _finish_images, args=(model,))
    timer.daemon = True
    _idle_timer = timer
    timer.start()


def start_service(name):
    """Démarre Ollama en processus détaché (commande fixe)."""
    if name != "ollama":
        raise ApiError(404, "Service inconnu.")
    if ollama_status()["online"]:
        return "Ollama est déjà démarré."
    if not installed()["ollama"]:
        raise ApiError(409, "Ollama est introuvable. Installez-le depuis ollama.com.")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / "ollama.log", "ab") as logf:
            subprocess.Popen([OLLAMA_EXE, "serve"], stdin=subprocess.DEVNULL, stdout=logf,  # noqa: S603
                             stderr=subprocess.STDOUT, creationflags=flags)
    except OSError:
        log.error("Démarrage du service impossible", exc_info=True)
        raise ApiError(500, "Impossible de démarrer le service.")
    return "Démarrage lancé."


def safe_project_dir(name):
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ApiError(400, "Nom de projet invalide (a-z, 0-9, - et _).")
    return PROJECTS_DIR / name


def safe_file_path(project_dir, rel):
    if not isinstance(rel, str) or not FILE_RE.match(rel) or ".." in rel or rel.endswith("/"):
        raise ApiError(400, "Nom de fichier invalide.")
    if Path(rel).suffix.lower() not in ALLOWED_EXT:
        raise ApiError(400, "Extension de fichier non autorisée.")
    target = (project_dir / rel).resolve()
    if project_dir.resolve() not in target.parents:
        raise ApiError(400, "Chemin invalide.")
    return target


def placeholder_svg(prompt, width=1024, height=768):
    words, lines, cur = prompt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 36:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    lines = lines[:6]
    text = "".join(
        f'<text x="50%" y="{46 + i * 5}%" text-anchor="middle" fill="#cbd5e1" '
        f'font-family="Segoe UI,sans-serif" font-size="30">{html.escape(l)}</text>'
        for i, l in enumerate(lines)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#312e81"/><stop offset="1" stop-color="#0f172a"/></linearGradient></defs>'
        f'<rect width="100%" height="100%" fill="url(#g)"/>'
        '<text x="50%" y="30%" text-anchor="middle" fill="#a5b4fc" font-family="Segoe UI,sans-serif" '
        'font-size="26" font-weight="700">VISUEL À GÉNÉRER (onglet Visuels)</text>'
        f"{text}</svg>"
    )
    return svg.encode("utf-8")


def unload_ollama_models():
    """Libère la RAM : décharge les modèles Ollama chargés (keep_alive=0) avant une génération d'image."""
    try:
        loaded = http_json(OLLAMA_URL + "/api/ps").get("models", [])
        for m in loaded:
            name = m.get("name")
            if isinstance(name, str) and name:
                http_json(OLLAMA_URL + "/api/generate", {"model": name, "keep_alive": 0}, 30)
        return len(loaded)
    except Exception:
        log.warning("Déchargement Ollama impossible", exc_info=True)
        return 0


# Progression de l'image en cours (lue par /api/status). Phases : load, sample, decode.
_progress = {}
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Barre de sd-cli : "  |=====>    |  3/8 -  12.34s/it - ETA 01:02" (vitesse en MB/s pendant le chargement).
_BAR_RE = re.compile(r"\|\s*(\d+)/(\d+)\s+-\s+([\d.]+)(s/it|it/s|MB/s|GB/s)(?:\s+-\s+ETA\s+([\d:]+))?")


def _set_progress(**fields):
    _progress.update(fields)


def _parse_sd_line(line):
    """Met à jour la progression à partir d'une ligne de sortie de sd-cli."""
    text = _ANSI_RE.sub("", line).strip()
    m = _BAR_RE.search(text)
    if m:
        step, steps, value, unit, eta = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), m.group(5)
        phase = "load" if unit.endswith("B/s") else "sample"
        _set_progress(phase=phase, step=step, steps=steps, speed=value + unit, eta=eta or "")
    elif "get_learned_condition completed" in text:
        _set_progress(phase="sample", step=0, steps=SD_STEPS, speed="", eta="")
    elif "sampling completed" in text:
        _set_progress(phase="decode", step=0, steps=0, speed="", eta="")


def _consume_sd_output(stream, logf):
    """Lit la sortie de sd-cli en continu (lignes séparées par \\r ou \\n), la journalise et suit la progression."""
    buf = b""
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        logf.write(chunk)
        buf = (buf + chunk)[-65536:]
        *lines, buf = re.split(rb"[\r\n]", buf)
        for raw in lines:
            if raw:
                _parse_sd_line(raw.decode("utf-8", "replace"))


# Génération en cours : processus sd-cli et fichier d'aperçu (écrasé par sd-cli à chaque étape).
_current = {"proc": None, "preview": None}
_cancel = threading.Event()
_PNG_SIG, _PNG_END = b"\x89PNG\r\n\x1a\n", b"IEND\xaeB`\x82"


def cancel_image():
    """Arrête la génération d'image en cours (amélioration du prompt ou processus sd-cli). Renvoie True si une image tournait."""
    if not _generating.is_set():
        return False
    _cancel.set()
    proc = _current["proc"]
    if proc is not None and proc.poll() is None:
        proc.kill()
    log.info("Génération d'image arrêtée par l'utilisateur")
    return True


def read_preview():
    """Dernier aperçu écrit par sd-cli, ou None s'il est absent ou en cours d'écriture (PNG incomplet)."""
    path = _current["preview"]
    try:
        data = path.read_bytes() if path else b""
    except OSError:
        return None
    return data if data.startswith(_PNG_SIG) and data.endswith(_PNG_END) else None


def run_sd(prompt, width, height, seed, device="cpu"):
    """Génère une image avec sd-cli (Z-Image Turbo) sur le périphérique donné et renvoie les octets PNG."""
    with tempfile.TemporaryDirectory(prefix="studio-img-") as tmp:
        out, preview = Path(tmp) / "out.png", Path(tmp) / "preview.png"
        # --preview proj : projection des latents en couleurs (sans décodage VAE), écrite à chaque étape.
        cmd = [str(SD_CLI), "--diffusion-model", str(SD_DIFFUSION), "--llm", str(SD_LLM), "--vae", str(SD_VAE),
               "-p", prompt.lstrip("- "), "--cfg-scale", "1.0", "--steps", str(SD_STEPS),
               "-W", str(width), "-H", str(height), "--vae-tiling",
               "--preview", "proj", "--preview-path", str(preview), "--preview-interval", "1", "-o", str(out)]
        if device != "cpu":
            cmd += ["--backend", device]
        elif gpu_status()["devices"]:
            cmd += ["--backend", "cpu"]  # build GPU : sans cette option sd-cli choisirait un GPU tout seul
        if seed is not None:
            cmd += ["-s", str(seed)]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        LOG_DIR.mkdir(exist_ok=True)
        started = _progress.get("started", time.monotonic())  # le chrono inclut l'amélioration du prompt
        _progress.clear()
        _progress["started"] = started
        _set_progress(phase="load", step=0, steps=0, speed="", eta="")
        with open(LOG_DIR / "sd.log", "ab") as logf:
            proc = subprocess.Popen(cmd, cwd=str(SD_DIR), stdin=subprocess.DEVNULL,  # noqa: S603
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
            _current.update(proc=proc, preview=preview)
            watchdog = threading.Timer(IMAGE_TIMEOUT, proc.kill)
            watchdog.start()
            try:
                _consume_sd_output(proc.stdout, logf)
                code = proc.wait()
            finally:
                watchdog.cancel()
                if proc.poll() is None:
                    proc.kill()
                _current.update(proc=None, preview=None)
        if _cancel.is_set():
            raise ApiError(409, "Génération arrêtée.")
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd[0])
        return out.read_bytes()


def find_improver(models):
    """Nom exact du modèle d'amélioration de prompts s'il est installé dans Ollama, sinon None."""
    return next((m for m in models if m == IMPROVE_MODEL or m.startswith(IMPROVE_MODEL + ":")), None)


def improve_prompt(model, prompt):
    """Réécrit le prompt avec le modèle Ollama d'amélioration, puis le décharge. Repli sur le prompt d'origine."""
    try:
        result = http_json(OLLAMA_URL + "/api/generate", {
            "model": model, "prompt": prompt, "stream": False, "keep_alive": 0,
            "options": {"num_ctx": 2048, "num_predict": 200}}, IMPROVE_TIMEOUT)
        text = str(result.get("response", "")).replace("<end_of_turn>", "")
        text = " ".join(text.split()).strip(" \"'`")[:IMPROVED_MAX]
        return text or prompt
    except Exception:
        log.warning("Amélioration du prompt impossible : prompt d'origine conservé", exc_info=True)
        return prompt


# Consigne remplaçant celle du modèle (réglée pour des prompts d'image) : ici on enrichit une demande d'application web.
APP_IMPROVE_SYSTEM = (
    "Rewrite the user's request for an AI web app builder so that it is clearer and more complete. "
    "Answer in the same language as the request. Keep every requirement of the original request; add only sensible "
    "details (main features, sections, content, visual style, responsive layout). The app uses plain HTML, CSS and "
    "JavaScript only, no frameworks. Maximum 150 words, as one short paragraph or a short list. "
    "Reply with the rewritten request only, without explanation or prefix.")
APP_IMPROVE_MODIFY = (
    " The request modifies an existing app: only clarify the requested change and do not invent new features.")
REQUEST_MAX = 6000
IMPROVED_REQUEST_MAX = 2000


def improve_request(body):
    """Enrichit la demande d'application de l'utilisateur avec le modèle d'amélioration, puis le décharge."""
    text = str(body.get("text", "")).strip()
    if not text or len(text) > REQUEST_MAX:
        raise ApiError(400, "Demande invalide (1 à 6000 caractères).")
    improver = find_improver(ollama_status()["models"])
    if not improver:
        raise ApiError(409, f"Le modèle {IMPROVE_MODEL} n'est pas installé dans Ollama.")
    system = APP_IMPROVE_SYSTEM + (APP_IMPROVE_MODIFY if body.get("modify") is True else "")
    if not _mem_lock.acquire(timeout=LOCK_WAIT):
        raise ApiError(409, "Une autre génération (projet ou image) est déjà en cours.")
    try:
        _cancel_idle_release()
        unload_ollama_models()  # un seul modèle en mémoire à la fois
        try:
            result = http_json(OLLAMA_URL + "/api/generate", {
                "model": improver, "system": system, "prompt": text, "stream": False, "keep_alive": 0,
                "options": {"num_ctx": 4096, "num_predict": 400, "temperature": 0.6}}, IMPROVE_TIMEOUT)
            lines = str(result.get("response", "")).replace("<end_of_turn>", "").splitlines()
            improved = "\n".join(l for l in (" ".join(l.split()) for l in lines) if l)[:IMPROVED_REQUEST_MAX].strip()
        except Exception:
            log.warning("Amélioration de la demande impossible : demande d'origine conservée", exc_info=True)
            improved = ""
        return {"text": improved or text, "model": improver, "changed": bool(improved) and improved != text}
    finally:
        _mem_lock.release()


def generate_image(body):
    prompt = str(body.get("prompt", "")).strip()
    if not prompt or len(prompt) > 2000:
        raise ApiError(400, "Prompt invalide (1 à 2000 caractères).")
    aspect = body.get("aspect", "768*576")
    if aspect not in ASPECTS:
        raise ApiError(400, "Paramètres d'image invalides.")
    seed = body.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31):
        raise ApiError(400, "Graine invalide.")
    width, height = (int(v) for v in aspect.split("*"))
    if body.get("placeholder") is True:
        # Placeholder SVG instantané : ni stable-diffusion.cpp ni Ollama ne sont sollicités.
        return placeholder_svg(prompt, width, height), "image/svg+xml", True, prompt
    if not installed()["images"]:
        raise ApiError(409, "Le moteur d'images (sd-cpp) est introuvable. Voir l'onglet Guide.")
    device = resolve_device(body.get("device"))
    model = body.get("model")  # modèle Ollama à recharger une fois les images terminées
    models = ollama_status()["models"]
    reload_model = model if isinstance(model, str) and model in models else None
    improver = find_improver(models) if body.get("improve") is True else None
    if not _mem_lock.acquire(timeout=LOCK_WAIT):
        raise ApiError(409, "Une autre génération (projet ou image) est déjà en cours.")
    try:
        _cancel_idle_release()  # image suivante d'un lot : Ollama reste déchargé
        unload_ollama_models()  # la RAM d'Ollama doit être libre avant de charger le modèle d'image
        _cancel.clear()
        _progress.clear()
        _progress["started"] = time.monotonic()
        _generating.set()
        try:
            if improver:
                _set_progress(phase="improve", step=0, steps=0, speed="", eta="")
                prompt = improve_prompt(improver, prompt)  # modèle déchargé (keep_alive 0) avant sd-cli
                if _cancel.is_set():
                    raise ApiError(409, "Génération arrêtée.")
            return run_sd(prompt, width, height, seed, device), "image/png", False, prompt
        except ApiError:
            raise
        except Exception:
            log.error("Échec de la génération d'image", exc_info=True)
            raise ApiError(502, "La génération d'image a échoué. Consultez logs/sd.log.")
        finally:
            _generating.clear()
    finally:
        _arm_idle_release(reload_model)  # sans nouvelle image sous 20 s : Ollama rechargé
        _mem_lock.release()


def save_project(body):
    pdir = safe_project_dir(body.get("name"))
    files = body.get("files")
    if not isinstance(files, dict) or not files or len(files) > 40:
        raise ApiError(400, "Liste de fichiers invalide.")
    targets = []
    for rel, content in files.items():
        target = safe_file_path(pdir, rel)
        if not isinstance(content, str) or len(content) > 2_000_000:
            raise ApiError(400, "Contenu de fichier invalide.")
        targets.append((rel, target, content))
    # Tout est validé avant la moindre écriture : pas de projet vide ni d'écriture partielle.
    for _, target, content in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return [rel for rel, _, _ in targets]


def list_project_files(pdir):
    if not pdir.is_dir():
        raise ApiError(404, "Projet introuvable.")
    return sorted(str(p.relative_to(pdir)).replace("\\", "/") for p in pdir.rglob("*") if p.is_file())


def list_projects():
    PROJECTS_DIR.mkdir(exist_ok=True)
    out = []
    for p in sorted(PROJECTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir() and NAME_RE.match(p.name):
            out.append({"name": p.name, "files": len(list_project_files(p)), "mtime": int(p.stat().st_mtime)})
    return out


# ---------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    server_version = "AppStudio/1.0"

    def log_message(self, fmt, *args):
        log.info(fmt % args)

    # -- réponses
    def _send(self, status, body, ctype, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ApiError(413, "Corps de requête invalide ou trop volumineux.")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            raise ApiError(400, "JSON invalide.")
        if not isinstance(data, dict):
            raise ApiError(400, "JSON invalide.")
        return data

    def _check_origin(self):
        allowed = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
        if self.headers.get("Host") not in allowed:
            raise ApiError(403, "Hôte non autorisé.")
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in allowed:
            raise ApiError(403, "Origine non autorisée.")

    def _guard(self, fn):
        try:
            self._check_origin()
            fn()
        except ApiError as e:
            self._json(e.status, {"error": e.message})
        except ConnectionError:  # client parti (BrokenPipe, Reset, Aborted) : rien à signaler
            pass
        except Exception:
            log.error("Erreur interne", exc_info=True)
            try:
                self._json(500, {"error": "Erreur interne du serveur."})
            except Exception:
                pass

    # -- routage
    def do_GET(self):
        self._guard(self._get)

    def do_POST(self):
        self._guard(self._post)

    def do_DELETE(self):
        self._guard(self._delete)

    def _get(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/image/preview":
            data = read_preview()
            if data is None:
                raise ApiError(404, "Aucun aperçu.")
            return self._send(200, data, "image/png", {"Cache-Control": "no-store"})
        if path == "/api/status":
            ollama = ollama_status()
            return self._json(200, {"ollama": ollama, "images": image_status(), "gpu": gpu_status(),
                                    "installed": installed(), "improve_model": find_improver(ollama["models"]),
                                    "ollama_url": OLLAMA_URL})
        if path == "/api/projects":
            return self._json(200, {"projects": list_projects()})
        m = re.fullmatch(r"/api/projects/([^/]+)/zip", path)
        if m:
            pdir = safe_project_dir(m.group(1))
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for rel in list_project_files(pdir):
                    if rel != "visuals.json":
                        zf.write(pdir / rel, rel)
            return self._send(200, buf.getvalue(), "application/zip",
                              {"Content-Disposition": f'attachment; filename="{pdir.name}.zip"'})
        m = re.fullmatch(r"/api/projects/([^/]+)", path)
        if m:
            pdir = safe_project_dir(m.group(1))
            files = {}
            for rel in list_project_files(pdir):
                if Path(rel).suffix.lower() in {".html", ".css", ".js", ".json", ".md", ".txt", ".svg"}:
                    files[rel] = (pdir / rel).read_text(encoding="utf-8", errors="replace")
            return self._json(200, {"name": pdir.name, "all": list_project_files(pdir), "files": files})
        m = re.fullmatch(r"/projects/([^/]+)/(.*)", path)
        if m:
            return self._serve_project(m.group(1), m.group(2) or "index.html")
        if path == "/":
            path = "/index.html"
        if path.startswith("/static/"):
            path = path[len("/static"):]
        return self._serve_file(STATIC_DIR, path.lstrip("/"), {})

    def _serve_project(self, name, rel):
        pdir = safe_project_dir(name)
        # CSP sandbox : le contenu généré ne peut pas accéder à l'origine du studio
        self._serve_file(pdir, rel, {"Content-Security-Policy": "sandbox allow-scripts allow-forms allow-modals"})

    def _serve_file(self, base, rel, extra):
        target = (base / rel).resolve()
        if base.resolve() not in target.parents or not target.is_file():
            raise ApiError(404, "Introuvable.")
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if data[:5] == b"<svg " or data[:4] == b"<svg":
            ctype = "image/svg+xml"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        extra = dict(extra)
        extra.setdefault("Cache-Control", "no-store")
        self._send(200, data, ctype, extra)

    def _post(self):
        path = urlparse(self.path).path
        m = re.fullmatch(r"/api/services/(ollama)/start", path)
        if m:
            return self._json(200, {"ok": True, "message": start_service(m.group(1))})
        if path == "/api/gpu/detect":
            return self._json(200, {"ok": True, "started": start_gpu_scan()})
        if path == "/api/image/cancel":
            return self._json(200, {"ok": True, "cancelled": cancel_image()})
        body = self._read_json()
        if path == "/api/improve":
            return self._json(200, improve_request(body))
        if path == "/api/chat":
            return self._chat(body)
        if path == "/api/image":
            return self._image(body)
        if path == "/api/projects":
            written = save_project(body)
            return self._json(200, {"ok": True, "name": body["name"], "written": written})
        raise ApiError(404, "Route inconnue.")

    def _delete(self):
        m = re.fullmatch(r"/api/projects/([^/]+)", urlparse(self.path).path)
        if not m:
            raise ApiError(404, "Route inconnue.")
        pdir = safe_project_dir(m.group(1))
        if not pdir.is_dir():
            raise ApiError(404, "Projet introuvable.")
        shutil.rmtree(pdir)
        self._json(200, {"ok": True})

    # -- Ollama (streaming NDJSON relayé tel quel)
    def _chat(self, body):
        model, messages = body.get("model"), body.get("messages")
        if not isinstance(model, str) or not model or len(model) > 120:
            raise ApiError(400, "Modèle invalide.")
        if not isinstance(messages, list) or not messages or len(messages) > 60:
            raise ApiError(400, "Messages invalides.")
        # keep_alive=0 : Ollama libère la RAM dès la fin de la réponse (le modèle d'image en a besoin).
        payload = {"model": model, "messages": messages, "stream": True, "keep_alive": 0,
                   "options": {"temperature": float(body.get("temperature", 0.4)), "num_ctx": 8192}}
        req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        # Verrou partagé avec la génération d'images : Ollama et le modèle d'image ne sont jamais chargés ensemble.
        # Attente courte : conception puis code s'enchaînent, la 2e requête peut arriver avant la libération du verrou.
        if not _mem_lock.acquire(timeout=LOCK_WAIT):
            raise ApiError(409, "Une génération (projet ou image) est déjà en cours.")
        try:
            _cancel_idle_release()
            try:
                resp = urllib.request.urlopen(req, timeout=CHAT_TIMEOUT)  # noqa: S310
            except (urllib.error.URLError, OSError):
                log.error("Ollama injoignable", exc_info=True)
                raise ApiError(502, "Ollama est injoignable. Vérifiez qu'il est démarré.")
            with resp:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                for line in resp:
                    self.wfile.write(line)
                    self.wfile.flush()
            self.close_connection = True
        finally:
            unload_ollama_models()  # projet terminé (ou interrompu) : Ollama libère la RAM
            _mem_lock.release()

    def _image(self, body):
        data, mime, placeholder, prompt_used = generate_image(body)
        saved = None
        if body.get("project") and body.get("file"):
            pdir = safe_project_dir(body["project"])
            target = safe_file_path(pdir, body["file"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            saved = body["file"]
        self._json(200, {"ok": True, "placeholder": placeholder, "mime": mime, "saved": saved,
                         "prompt_used": prompt_used, "data": base64.b64encode(data).decode("ascii")})


class StudioServer(ThreadingHTTPServer):
    # Windows : SO_REUSEADDR laisse plusieurs processus écouter sur le même port, et une ancienne
    # instance (code périmé) peut alors répondre à la place de la nouvelle. On préfère échouer.
    allow_reuse_address = False


def main():
    PROJECTS_DIR.mkdir(exist_ok=True)
    start_gpu_scan()  # en arrière-plan : n'empêche pas le démarrage
    server = StudioServer((HOST, PORT), Handler)
    print(f"App Studio : http://{HOST}:{PORT}  (Ollama: {OLLAMA_URL} | images: sd-cpp)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
