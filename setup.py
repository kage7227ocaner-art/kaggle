import os
import re
import sys
import time
import shutil
import subprocess
import urllib.request
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

WORK_DIR = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "qwen38"

MODEL_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
MODEL_FILE = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
MODEL_NAME = "qwen38-27b"

MODEL_PATH = MODEL_DIR / MODEL_FILE
MODELFILE = WORK_DIR / "Modelfile"

OLLAMA_HOST = "127.0.0.1:11434"
OLLAMA_URL = f"http://{OLLAMA_HOST}"

OLLAMA_LOG = WORK_DIR / "ollama.log"
CLOUDFLARED_LOG = WORK_DIR / "cloudflared.log"

CLOUDFLARED_DEB = WORK_DIR / "cloudflared-linux-amd64.deb"

# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

# ============================================================
# HELPERS
# ============================================================

def log(message):
    print(f"[SETUP] {message}", flush=True)

def command_exists(command):
    return shutil.which(command) is not None

def run(command, check=True, capture=False):
    command = [str(x) for x in command]

    log("$ " + " ".join(command))

    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
        env=os.environ.copy(),
    )

def api_ready():
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags",
            timeout=5,
        ) as response:
            return response.status == 200
    except Exception:
        return False

def get_models():
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags",
            timeout=5,
        ) as response:
            return response.read().decode(
                "utf-8",
                errors="replace",
            )
    except Exception:
        return ""

# ============================================================
# CREATE DIRECTORIES
# ============================================================

WORK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

log(f"Working directory: {WORK_DIR}")
log(f"Model directory: {MODEL_DIR}")

# ============================================================
# GPU CHECK
# ============================================================

log("Checking NVIDIA GPU...")

if command_exists("nvidia-smi"):
    run(
        ["nvidia-smi"],
        check=False,
    )
else:
    log("WARNING: nvidia-smi not found.")

# ============================================================
# INSTALL SYSTEM PACKAGES
# ============================================================

log("Updating apt package list...")

if os.geteuid() == 0:
    SUDO = []
else:
    SUDO = ["sudo"]

run(
    SUDO + ["apt-get", "update"],
)

run(
    SUDO + [
        "apt-get",
        "install",
        "-y",
        "curl",
        "wget",
        "zstd",
        "ca-certificates",
    ],
)

# ============================================================
# INSTALL OLLAMA
# ============================================================

if command_exists("ollama"):
    log("Ollama already installed.")

else:
    log("Installing Ollama...")

    run(
        [
            "bash",
            "-c",
            "curl -fsSL https://ollama.com/install.sh | sh",
        ]
    )

if not command_exists("ollama"):
    log("ERROR: Ollama installation failed.")
    sys.exit(1)

# ============================================================
# START OLLAMA
# ============================================================

if api_ready():
    log("Ollama is already running.")

else:
    log("Starting Ollama...")

    ollama_log = open(
        OLLAMA_LOG,
        "a",
        buffering=1,
    )

    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=ollama_log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )

    log(f"Ollama PID: {ollama_process.pid}")

    log("Waiting for Ollama API...")

    ready = False

    for _ in range(30):
        if api_ready():
            ready = True
            break

        time.sleep(2)

    if not ready:
        log("ERROR: Ollama API did not start.")

        if OLLAMA_LOG.exists():
            print(
                OLLAMA_LOG.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-5000:]
            )

        sys.exit(1)

log("Ollama API is ready.")

# ============================================================
# INSTALL HUGGING FACE HUB
# ============================================================

log("Installing Hugging Face Hub...")

run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "huggingface_hub",
    ]
)

# ============================================================
# FIND HF COMMAND
# ============================================================

HF = shutil.which("hf")

if HF is None:
    candidate = Path(sys.executable).parent / "hf"

    if candidate.exists():
        HF = str(candidate)

if HF is None:
    candidate = Path.home() / ".local" / "bin" / "hf"

    if candidate.exists():
        HF = str(candidate)

if HF is None:
    log("ERROR: Hugging Face CLI was not found.")
    sys.exit(1)

log(f"Hugging Face CLI: {HF}")

# ============================================================
# DOWNLOAD GGUF
# ============================================================

if MODEL_PATH.exists():
    size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)

    log(
        f"GGUF already exists "
        f"({size_gb:.2f} GB)."
    )

else:
    log("Downloading GGUF model...")
    log(f"Repository: {MODEL_REPO}")
    log(f"File: {MODEL_FILE}")

    run(
        [
            HF,
            "download",
            MODEL_REPO,
            MODEL_FILE,
            "--local-dir",
            MODEL_DIR,
        ]
    )

if not MODEL_PATH.exists():
    log("ERROR: GGUF file was not downloaded.")
    sys.exit(1)

log(f"GGUF ready: {MODEL_PATH}")

# ============================================================
# CREATE MODELFILE
# ============================================================

log("Creating Modelfile...")

MODELFILE.write_text(
    f"""FROM {MODEL_PATH}

PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_p 0.9
""",
    encoding="utf-8",
)

log(f"Modelfile: {MODELFILE}")

# ============================================================
# CREATE OLLAMA MODEL
# ============================================================

models = get_models()

if (
    f'"name":"{MODEL_NAME}"' in models
    or f'"name": "{MODEL_NAME}"' in models
):
    log(f"Ollama model already exists: {MODEL_NAME}")

else:
    log(f"Creating Ollama model: {MODEL_NAME}")

    run(
        [
            "ollama",
            "create",
            MODEL_NAME,
            "-f",
            MODELFILE,
        ]
    )

log("Ollama model created.")

# ============================================================
# VERIFY MODEL
# ============================================================

models = get_models()

if MODEL_NAME in models:
    log(f"MODEL READY: {MODEL_NAME}")
else:
    log(
        "WARNING: Model was not detected "
        "through /api/tags."
    )

# ============================================================
# INSTALL CLOUDFLARED
# ============================================================

if command_exists("cloudflared"):
    log("cloudflared already installed.")

else:
    log("Downloading cloudflared...")

    if not CLOUDFLARED_DEB.exists():
        run(
            [
                "wget",
                "-q",
                "-O",
                CLOUDFLARED_DEB,
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
            ]
        )

    log("Installing cloudflared...")

    run(
        SUDO + [
            "dpkg",
            "-i",
            CLOUDFLARED_DEB,
        ],
        check=False,
    )

    run(
        SUDO + [
            "apt-get",
            "-f",
            "install",
            "-y",
        ],
        check=False,
    )

if not command_exists("cloudflared"):
    log("ERROR: cloudflared installation failed.")
    sys.exit(1)

# ============================================================
# START CLOUDFLARE QUICK TUNNEL
# ============================================================

log("Starting Cloudflare Quick Tunnel...")

try:
    CLOUDFLARED_LOG.unlink()
except FileNotFoundError:
    pass

cloudflared_log = open(
    CLOUDFLARED_LOG,
    "w",
    buffering=1,
)

cloudflared_process = subprocess.Popen(
    [
        "cloudflared",
        "tunnel",
        "--no-autoupdate",
        "--protocol",
        "http2",
        "--url",
        "http://127.0.0.1:11434",
        "--http-host-header",
        "127.0.0.1:11434",
    ],
    stdout=cloudflared_log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

log(
    f"cloudflared PID: "
    f"{cloudflared_process.pid}"
)

# ============================================================
# WAIT FOR CLOUDFLARE URL
# ============================================================

log("Waiting for Cloudflare URL...")

PUBLIC_URL = None

for _ in range(30):
    time.sleep(2)

    if not CLOUDFLARED_LOG.exists():
        continue

    text = CLOUDFLARED_LOG.read_text(
        encoding="utf-8",
        errors="replace",
    )

    matches = re.findall(
        r"https://[a-zA-Z0-9-]+\.trycloudflare\.com",
        text,
    )

    if matches:
        PUBLIC_URL = matches[-1]
        break

# ============================================================
# FINAL STATUS
# ============================================================

print("")
print("=" * 70)
print("KAGGLE OLLAMA SERVER")
print("=" * 70)

print(f"Model : {MODEL_NAME}")
print(f"GGUF : {MODEL_PATH}")
print(f"Local API : {OLLAMA_URL}")

if PUBLIC_URL:
    print(f"Public URL : {PUBLIC_URL}")
    print(f"Public API : {PUBLIC_URL}/api")
    print(f"API Tags : {PUBLIC_URL}/api/tags")
else:
    print("Public URL : NOT FOUND")

print("=" * 70)
print("")

# ============================================================
# SHOW CLOUDFLARED LOG IF URL WAS NOT FOUND
# ============================================================

# if not PUBLIC_URL:
#     log("Cloudflare URL was not detected.")

#     if CLOUDFLARED_LOG.exists():
#         print(
#             CLOUDFLARED_LOG.read_text(
#                 encoding="utf-8",
#                 errors="replace",
#             )[-5000:]
#         )


if PUBLIC_URL:
    PUBLIC_API = f"{PUBLIC_URL}/api/tags"

    log(f"Testing public API: {PUBLIC_API}")

    try:
        request = urllib.request.Request(
            PUBLIC_API,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=15,
        ) as response:
            body = response.read().decode(
                "utf-8",
                errors="replace",
            )

            print("")
            print("=" * 70)
            print("PUBLIC OLLAMA API: OK")
            print("=" * 70)
            print(f"URL: {PUBLIC_API}")
            print(body)
            print("=" * 70)
            print("")

    except Exception as error:
        print("")
        print("=" * 70)
        print("PUBLIC OLLAMA API: FAILED")
        print("=" * 70)
        print(error)
        print("")
        print("Cloudflared log:")
        print(
            CLOUDFLARED_LOG.read_text(
                encoding="utf-8",
                errors="replace",
            )[-5000:]
        )
        print("=" * 70)
        print("")

# ============================================================
# KEEP KAGGLE SESSION ALIVE
# ============================================================

log("Server is running.")
log("Keep this Kaggle session active.")

try:
    while True:
        time.sleep(60)

except KeyboardInterrupt:
    log("setup.py stopped.")
