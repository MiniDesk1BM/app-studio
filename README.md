# App Studio - Ollama x Z-Image Turbo

Interface locale pour créer des applications web à partir d'une description :

1. **Conception** (facultative) : un premier modèle Ollama rédige un brief sans code ; choisissez « Aucune » pour la sauter.
2. **Code** : un second modèle Ollama, au choix, implémente le brief ; les fichiers HTML/CSS/JS sont extraits et enregistrés dans `projects/<nom>/`.
3. **Visuels** : des placeholders SVG sont créés ; chaque visuel est généré à la demande avec Z-Image Turbo via stable-diffusion.cpp (onglet Visuels). Un seul gros modèle est en mémoire à la fois : Ollama (`keep_alive: 0`) est déchargé dès que le projet est généré, et avant chaque image. Une fois le lot d'images terminé (20 s sans nouvelle image), le modèle de code est rechargé. Une seule génération à la fois (erreur 409 sinon).
4. **Aperçu** : l'application s'exécute dans un iframe isolé (sandbox + CSP), avec vues bureau/tablette/mobile.

## Démarrage

```
start.bat            (ou : python server.py)
```

Ouvrir http://127.0.0.1:8080. Aucune dépendance Python à installer (bibliothèque standard).

## Composants

| Composant | Emplacement | Variable |
|---|---|---|
| Ollama | `http://127.0.0.1:11434` | `OLLAMA_URL`, `OLLAMA_EXE` |
| stable-diffusion.cpp | `./sd-cpp` | `SD_CPP_DIR` |
| App Studio | `http://127.0.0.1:8080` | `STUDIO_PORT` |

Le panneau **Services** de l'interface affiche l'état d'Ollama (avec « Démarrer » s'il est arrêté) et celui du moteur d'images.

- Ollama : `ollama pull qwen2.5-coder:7b` (un modèle « coder » est recommandé).
- Amélioration des prompts (facultatif) : `ollama pull gnokit/improve-prompt` (variable `IMPROVE_MODEL`). La case « Améliorer les prompts » de l'onglet Visuels réécrit chaque description avant la génération ; le modèle est déchargé juste après, puis `sd-cli` démarre. Le prompt effectivement utilisé est affiché sous la description. Si l'amélioration échoue, le prompt d'origine est conservé. La case « Améliorer ma demande » (zone de saisie) enrichit de la même façon la demande d'application avant la conception et le code (route `POST /api/improve`).
- Images : `sd-cli.exe` (build CPU de [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)) est lancé à la demande, une fois par image, puis s'arrête. Aucun GPU requis, mais comptez plusieurs minutes par image sur processeur. Les logs sont dans `logs/sd.log`.

Fichiers attendus (le bouton de génération reste désactivé s'il en manque un) :

```
sd-cpp/sd-cli.exe
sd-cpp/models/z_image_turbo-Q4_K.gguf               (leejet/Z-Image-Turbo-GGUF)
sd-cpp/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf    (unsloth/Qwen3-4B-Instruct-2507-GGUF)
sd-cpp/models/ae.safetensors                        (Comfy-Org/z_image_turbo, split_files/vae)
```

Formats d'image acceptés (largeur*hauteur) : `768*576`, `1024*576`, `768*768`, `576*768`, `576*1024`.

## Sécurité

- Serveur lié à 127.0.0.1, contrôle des en-têtes Host/Origin.
- Noms de projets et de fichiers validés par liste d'autorisation, extensions restreintes.
- `sd-cli` est lancé avec une liste d'arguments fixe (aucun shell) ; le prompt est passé comme valeur d'option après nettoyage des tirets initiaux.
- Contenu généré servi avec `Content-Security-Policy: sandbox` et affiché dans un iframe sandboxé.
- Interface avec CSP stricte, sans `innerHTML` sur du contenu généré.
