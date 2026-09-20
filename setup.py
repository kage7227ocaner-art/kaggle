import os
import re
import sys
import json
import time
import signal
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# ============================================================

# CONFIGURATION

# ============================================================

MODEL_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
MODEL_FILE = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
OLLAMA_MODEL = "qwen38-27b"

WORK_DIR = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "qwen38"
MODEL_PATH = MODEL_DIR / MODEL_FILE

MODELFILE_PATH = WORK_DIR / "Modelfile"

OLLAMA_HOST = "127.0.0.1:11434"
OLLAMA_API = f"http://{OLLAMA_HOST}"

OLLAMA_LOG = WORK_DIR / "ollama.log"
CLOUDFLARED_LOG = WORK_DIR / "cloudflared.log"

CLOUDFLARED_DEB = WORK_DIR / "cloudflared-linux-amd64.deb"

OLLAMA_PID_FILE = WORK_DIR / "ollama.pid"
CLOUDFLARED_PID_FILE = WORK_DIR / "cloudflared.pid"

# GPU configuration

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Ollama configuration

os.environ["OLLAMA_HOST"] = OLLAMA_HOST
os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

# ============================================================

# HELPERS

# ============================================================

def log(message=""):
print(f"[SETUP] {message}", flush=True)

def command_exists(command):
return shutil.which(command) is not None

def run(command, check=True, capture=False, env=None):
command = [str(x) for x in command]


log("$ " + " ".join(command))

return subprocess.run(
    command,
    check=check,
    text=True,
    capture_output=capture,
    env=env,
)


def process_alive(pid):
if not pid:
return False


try:
    os.kill(int(pid), 0)
    return True
except (ProcessLookupError, ValueError):
    return False
except PermissionError:
    return True


def read_pid_file(path):
if not path.exists():
return None


try:
    return int(path.read_text().strip())
except Exception:
    return None


def write_pid_file(path, pid):
path.write_text(str(pid))

def http_get(url, timeout=10):
try:
request = urllib.request.Request(
url,
headers={
"User-Agent": "kaggle-ollama-setup"
},
)


    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body

except urllib.error.HTTPError as error:
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""

    return error.code, body

except Exception as error:
    return None, str(error)


def wait_for_ollama(timeout=60):
log("Checking Ollama API...")


start = time.time()

while time.time() - start < timeout:
    status, body = http_get(f"{OLLAMA_API}/api/tags", timeout=3)

    if status == 200:
        log("Ollama API is ready.")
        return True

    time.sleep(2)

return False


def get_ollama_models():
status, body = http_get(f"{OLLAMA_API}/api/tags", timeout=5)


if status != 200:
    return []

try:
    data = json.loads(body)
    return [
        model.get("name", "")
        for model in data.get("models", [])
    ]
except Exception:
    return []


# ============================================================

# CREATE DIRECTORIES

# ============================================================

log("Creating required directories...")

WORK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

log(f"Working directory : {WORK_DIR}")
log(f"Model directory   : {MODEL_DIR}")

# ============================================================

# SYSTEM INFORMATION

# ============================================================

log("Checking GPU...")

if command_exists("nvidia-smi"):
run(["nvidia-smi"], check=False)
else:
log("WARNING: nvidia-smi was not found.")

# ============================================================

# APT / SYSTEM DEPENDENCIES

# ============================================================

log("Installing system dependencies...")

if os.geteuid() == 0:
SUDO = []
else:
SUDO = ["sudo"]

run(
SUDO + ["apt-get", "update"],
check=True,
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
"git",
],
check=True,
)

# ============================================================

# INSTALL OLLAMA

# ============================================================

if command_exists("ollama"):
log("Ollama is already installed.")


try:
    result = run(
        ["ollama", "--version"],
        check=False,
        capture=True,
    )

    output = (result.stdout or result.stderr).strip()

    if output:
        log(output)

except Exception:
    pass


else:
log("Installing Ollama...")


run(
    [
        "bash",
        "-c",
        "curl -fsSL https://ollama.com/install.sh | sh",
    ],
    check=True,
)


if not command_exists("ollama"):
log("ERROR: Ollama installation failed.")
sys.exit(1)

# ============================================================

# START OLLAMA

# ============================================================

ollama_pid = read_pid_file(OLLAMA_PID_FILE)

if ollama_pid and process_alive(ollama_pid):
log(f"Ollama process already running. PID={ollama_pid}")

elif wait_for_ollama(timeout=3):
log("Ollama is already running.")

else:
log("Starting Ollama server...")


ollama_log_file = open(
    OLLAMA_LOG,
    "a",
    buffering=1,
)

ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=ollama_log_file,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
    start_new_session=True,
)

write_pid_file(
    OLLAMA_PID_FILE,
    ollama_process.pid,
)

log(f"Ollama started. PID={ollama_process.pid}")

if not wait_for_ollama(timeout=60):
    log("ERROR: Ollama API did not become ready.")

    log("Last Ollama log:")
    try:
        print(OLLAMA_LOG.read_text()[-5000:])
    except Exception:
        pass

    sys.exit(1)


# ============================================================

# INSTALL HUGGING FACE HUB

# ============================================================

log("Installing/updating Hugging Face Hub...")

run(
[
sys.executable,
"-m",
"pip",
"install",
"-q",
"-U",
"huggingface_hub",
],
check=True,
)

# ============================================================

# FIND HF CLI

# ============================================================

HF_COMMAND = shutil.which("hf")

if HF_COMMAND is None:
possible_hf = [
Path(sys.executable).parent / "hf",
Path.home() / ".local" / "bin" / "hf",
]


for candidate in possible_hf:
    if candidate.exists():
        HF_COMMAND = str(candidate)
        break


if HF_COMMAND is None:
log("ERROR: Hugging Face CLI 'hf' was not found.")
sys.exit(1)

log(f"Hugging Face CLI: {HF_COMMAND}")

# ============================================================

# DOWNLOAD GGUF

# ============================================================

if MODEL_PATH.exists():
size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)


log(
    f"GGUF already exists: "
    f"{MODEL_PATH} "
    f"({size_gb:.2f} GB)"
)


else:
log("GGUF model not found.")
log("Downloading model from Hugging Face...")
log(f"Repository : {MODEL_REPO}")
log(f"File       : {MODEL_FILE}")


run(
    [
        HF_COMMAND,
        "download",
        MODEL_REPO,
        MODEL_FILE,
        "--local-dir",
        str(MODEL_DIR),
    ],
    check=True,
)


if not MODEL_PATH.exists():
log("ERROR: GGUF download failed.")
sys.exit(1)

# ============================================================

# CREATE OLLAMA MODELFILE

# ============================================================

log("Creating Ollama Modelfile...")

MODELFILE_CONTENT = f"""FROM {MODEL_PATH}

PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

MODELFILE_PATH.write_text(
MODELFILE_CONTENT,
encoding="utf-8",
)

log(f"Modelfile: {MODELFILE_PATH}")

# ============================================================

# CREATE OLLAMA MODEL

# ============================================================

log("Checking whether Ollama model already exists...")

existing_models = get_ollama_models()

model_exists = any(
name == OLLAMA_MODEL
or name == f"{OLLAMA_MODEL}:latest"
for name in existing_models
)

if model_exists:
log(
f"Ollama model already exists: "
f"{OLLAMA_MODEL}"
)

else:
log(f"Creating Ollama model: {OLLAMA_MODEL}")


run(
    [
        "ollama",
        "create",
        OLLAMA_MODEL,
        "-f",
        str(MODELFILE_PATH),
    ],
    check=True,
)

log("Ollama model created successfully.")


# ============================================================

# VERIFY LOCAL OLLAMA MODEL

# ============================================================

log("Verifying local Ollama model...")

models = get_ollama_models()

if not any(
name == OLLAMA_MODEL
or name == f"{OLLAMA_MODEL}:latest"
for name in models
):
log("WARNING: Model was not found through /api/tags.")


log("Current Ollama models:")
for model in models:
    print(" -", model)


else:
log(f"Model ready: {OLLAMA_MODEL}")

# ============================================================

# INSTALL CLOUDFLARED

# ============================================================

if command_exists("cloudflared"):
log("cloudflared is already installed.")

else:
log("Installing cloudflared...")


if not CLOUDFLARED_DEB.exists():
    run(
        [
            "wget",
            "-q",
            "--show-progress",
            "-O",
            str(CLOUDFLARED_DEB),
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
        ],
        check=True,
    )

run(
    SUDO + [
        "dpkg",
        "-i",
        str(CLOUDFLARED_DEB),
    ],
    check=False,
)

# Fix missing dependencies if necessary.
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

cloudflared_pid = read_pid_file(CLOUDFLARED_PID_FILE)

cloudflared_running = (
cloudflared_pid is not None
and process_alive(cloudflared_pid)
)

if cloudflared_running:
log(
f"cloudflared tunnel already running. "
f"PID={cloudflared_pid}"
)

else:
log("Starting Cloudflare Quick Tunnel...")


# Clear old URL/log information.
try:
    CLOUDFLARED_LOG.unlink()
except FileNotFoundError:
    pass

cloudflared_log_file = open(
    CLOUDFLARED_LOG,
    "w",
    buffering=1,
)

cloudflared_process = subprocess.Popen(
    [
        "cloudflared",
        "tunnel",
        "--no-autoupdate",
        "--url",
        OLLAMA_API,
    ],
    stdout=cloudflared_log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

write_pid_file(
    CLOUDFLARED_PID_FILE,
    cloudflared_process.pid,
)

cloudflared_pid = cloudflared_process.pid

log(
    f"cloudflared started. "
    f"PID={cloudflared_pid}"
)


# ============================================================

# FIND PUBLIC CLOUDFLARE URL

# ============================================================

log("Waiting for Cloudflare public URL...")

public_url = None

for _ in range(60):
time.sleep(2)


if not CLOUDFLARED_LOG.exists():
    continue

try:
    log_content = CLOUDFLARED_LOG.read_text(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    continue

matches = re.findall(
    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com",
    log_content,
)

if matches:
    public_url = matches[-1]
    break


if public_url:
log("")
log("=" * 60)
log("CLOUDFLARE PUBLIC URL")
log("=" * 60)
print(public_url)
log("=" * 60)
log("")

else:
log("WARNING: Cloudflare public URL was not detected.")


if CLOUDFLARED_LOG.exists():
    log("cloudflared log:")
    print(
        CLOUDFLARED_LOG.read_text(
            encoding="utf-8",
            errors="replace",
        )[-5000:]
    )


# ============================================================

# TEST LOCAL OLLAMA API

# ============================================================

log("Testing local Ollama API...")

status, body = http_get(
f"{OLLAMA_API}/api/tags",
timeout=10,
)

if status == 200:
log("LOCAL API TEST: OK")
else:
log(
f"LOCAL API TEST FAILED "
f"(HTTP {status})"
)

# ============================================================

# TEST PUBLIC OLLAMA API

# ============================================================

if public_url:
log("Testing public Ollama API...")


public_api_url = (
    f"{public_url}/api/tags"
)

public_status = None

for attempt in range(10):
    public_status, public_body = http_get(
        public_api_url,
        timeout=15,
    )

    if public_status == 200:
        break

    log(
        f"Public API attempt "
        f"{attempt + 1}/10 failed: "
        f"HTTP {public_status}"
    )

    time.sleep(3)

if public_status == 200:
    log("PUBLIC API TEST: OK")

    print("")
    print("Public Ollama API:")
    print(public_api_url)
    print("")

else:
    log(
        "PUBLIC API TEST FAILED."
    )

    log(
        "This usually means the tunnel "
        "cannot reach the Ollama origin."
    )

    log("")
    log("Ollama log:")
    try:
        print(
            OLLAMA_LOG.read_text(
                encoding="utf-8",
                errors="replace",
            )[-3000:]
        )
    except Exception:
        pass

    log("")
    log("cloudflared log:")
    try:
        print(
            CLOUDFLARED_LOG.read_text(
                encoding="utf-8",
                errors="replace",
            )[-5000:]
        )
    except Exception:
        pass


# ============================================================

# FINAL STATUS

# ============================================================

print("")
print("=" * 70)
print("KAGGLE OLLAMA SETUP COMPLETE")
print("=" * 70)

print(f"Model        : {OLLAMA_MODEL}")
print(f"GGUF         : {MODEL_PATH}")
print(f"Ollama API   : {OLLAMA_API}")

if public_url:
print(f"Public URL   : {public_url}")
print(f"Public API   : {public_url}/api")
print(f"Tags         : {public_url}/api/tags")
else:
print("Public URL   : NOT DETECTED")

print("")
print("Processes:")
print(f"Ollama PID   : {read_pid_file(OLLAMA_PID_FILE)}")
print(
f"Cloudflared  : "
f"{read_pid_file(CLOUDFLARED_PID_FILE)}"
)

print("=" * 70)
print("")

# ============================================================

# KEEP NOTEBOOK PROCESS ALIVE

# ============================================================

log(
"Keeping Kaggle session alive. "
"Press Stop/Interrupt to terminate this script."
)

try:
while True:
time.sleep(30)


    # Basic process monitoring.
    ollama_pid = read_pid_file(OLLAMA_PID_FILE)
    cloudflared_pid = read_pid_file(
        CLOUDFLARED_PID_FILE
    )

    if ollama_pid and not process_alive(ollama_pid):
        log(
            "WARNING: Ollama process is no longer running."
        )

    if cloudflared_pid and not process_alive(
        cloudflared_pid
    ):
        log(
            "WARNING: cloudflared process is no longer running."
        )


except KeyboardInterrupt:
log("Setup process interrupted.")

finally:
log("Exiting setup.py.")
log(
"Background processes may remain active "
"until the Kaggle session is terminated."
)
