"""Télécharge les modèles Z-Image Turbo (GGUF) dans sd-cpp/models (bibliothèque standard uniquement).

Usage : python download_models.py
 - saute les fichiers déjà présents ;
 - reprend un téléchargement interrompu (fichier .part) ;
 - HTTPS uniquement, domaine huggingface.co (liste d'autorisation).
"""
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SD_DIR = Path(os.environ.get("SD_CPP_DIR", Path(__file__).resolve().parent / "sd-cpp"))
MODELS_DIR = SD_DIR / "models"
ALLOWED_HOSTS = {"huggingface.co"}  # les redirections vers le CDN sont suivies par urllib après contrôle de l'URL d'origine
TIMEOUT = 60
CHUNK = 1024 * 1024

MODELS = {
    "z_image_turbo-Q4_K.gguf":
        "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K.gguf",
    "Qwen3-4B-Instruct-2507-Q4_K_M.gguf":
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "ae.safetensors":
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
}


def download(name, url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL non autorisée : {url}")
    dest = MODELS_DIR / name
    if dest.is_file():
        print(f"[ok]  {name} déjà présent")
        return
    part = dest.with_name(name + ".part")
    done = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "app-studio-downloader"}
    if done:
        headers["Range"] = f"bytes={done}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT) as resp:  # noqa: S310
        if done and resp.status != 206:  # le serveur ignore Range : on repart de zéro
            done = 0
        length = int(resp.headers.get("Content-Length") or 0)
        total = done + length if length else 0
        with open(part, "ab" if done else "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r[...] {name} : {done / 1e6:,.0f} / {total / 1e6:,.0f} Mo ({done * 100 // total} %)",
                          end="", flush=True)
    if total and done != total:
        raise IOError(f"téléchargement incomplet ({done}/{total} octets) : relancez le script pour reprendre")
    part.replace(dest)
    print(f"\r[ok]  {name} téléchargé ({done / 1e6:,.0f} Mo)" + " " * 20)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, url in MODELS.items():
        try:
            download(name, url)
        except Exception as e:  # noqa: BLE001
            failed = True
            print(f"\n[err] {name} : {e}")
    if not (SD_DIR / "sd-cli.exe").is_file():
        print("[!]   sd-cli.exe absent : voir README (release stable-diffusion.cpp).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
