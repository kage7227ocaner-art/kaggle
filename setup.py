#!/usr/bin/env python3

import os
import re
import sys
import time
import signal
import shutil
import socket
import subprocess
import urllib.request
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

CLOUDFLARED_DEB = WORK_DIR / "cloudflared-linux-amd64.deb"
CLOUDFLARED_LOG = WORK_DIR / "cloudflared.log"

# 2x Tesla T4
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Ollama configuration
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

# ============================================================
# HELPERS
# ============================================================

processes = []


def run(command, check=True, capture=False):
    """Run shell command."""
    print("\n$ " + " ".join(map(str, command)))

    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def command_exists(command):
    return shutil.which(command) is not None


def wait_for_port(host, port, timeout=60):
    """Wait until TCP port becomes available."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)

    return False


def http_get(url, timeout=10):
    """Simple HTTP GET without requiring requests."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except Exception as e:
        return None, str(e)


def terminate_processes():
    """Clean up child processes."""
    for process in processes:
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass


def signal_handler(signum, frame):
    print("\nStopping processes...")
    terminate_processes()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# 1. SYSTEM UPDATE
# ============================================================

print("=" * 70)
print("1/9 - Updating Ubuntu package lists")
print("=" * 70)

run(["apt-get", "update"])


# ============================================================
# 2. INSTALL SYSTEM DEPENDENCIES
# ============================================================

print("=" * 70)
print("2/9 - Installing system dependencies")
print("=" * 70)

run([
    "apt-get",
    "install",
    "-y",
    "zstd",
    "curl",
    "wget",
    "ca-certificates",
])


# ============================================================
# 3. INSTALL OLLAMA
# ============================================================

print("=" * 70)
print("3/9 - Installing Ollama")
print("=" * 70)

if command_exists("ollama"):
    print("Ollama is already installed.")
else:
    install_script = WORK_DIR / "install-ollama.sh"

    with open(install_script, "wb") as f:
        response = urllib.request.urlopen(
            "https://ollama.com/install.sh",
            timeout=60,
        )
        f.write(response.read())

    os.chmod(install_script, 0o755)

    run(["bash", str(install_script)])

print("\nOllama version:")
run(["ollama", "--version"])


# ============================================================
# 4. START OLLAMA SERVER
# ============================================================

print("=" * 70)
print("4/9 - Starting Ollama server")
print("=" * 70)

# Avoid starting a second Ollama server.
status, body = http_get(f"{OLLAMA_API}/api/tags")

if status == 200:
    print("Ollama is already running.")
else:
    print("Starting Ollama in background...")

    ollama_log = open(WORK_DIR / "ollama.log", "a")

    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=ollama_log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )

    processes.append(ollama_process)

    print("Ollama PID:", ollama_process.pid)

    if not wait_for_port("127.0.0.1", 11434, timeout=60):
        print("\nERROR: Ollama did not open port 11434.")
        print("Check:")
        print(WORK_DIR / "ollama.log")
        sys.exit(1)


# ============================================================
# 5. TEST OLLAMA API
# ============================================================

print("=" * 70)
print("5/9 - Testing Ollama API")
print("=" * 70)

status, body = http_get(f"{OLLAMA_API}/api/tags")

if status != 200:
    print("ERROR: Ollama API is not responding.")
    print("Status:", status)
    print("Response:", body)
    sys.exit(1)

print("Ollama API: OK")
print(body[:1000])


# ============================================================
# 6. INSTALL HUGGING FACE HUB
# ============================================================

print("=" * 70)
print("6/9 - Installing Hugging Face Hub")
print("=" * 70)

run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-U",
    "huggingface_hub",
])

# Locate hf executable
hf_command = shutil.which("hf")

if not hf_command:
    candidate = Path(sys.executable).parent / "hf"

    if candidate.exists():
        hf_command = str(candidate)

if not hf_command:
    print("ERROR: 'hf' command was not found.")
    sys.exit(1)

print("Hugging Face CLI:", hf_command)


# ============================================================
# 7. DOWNLOAD GGUF
# ============================================================

print("=" * 70)
print("7/9 - Downloading GGUF model")
print("=" * 70)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_PATH.exists():
    size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)

    print(
        f"Model already exists: {MODEL_PATH}"
    )
    print(
        f"Size: {size_gb:.2f} GB"
    )
else:
    print("Downloading:")
    print(f"Repository : {MODEL_REPO}")
    print(f"File       : {MODEL_FILE}")
    print(f"Destination: {MODEL_DIR}")

    run([
        hf_command,
        "download",
        MODEL_REPO,
        MODEL_FILE,
        "--local-dir",
        str(MODEL_DIR),
    ])

if not MODEL_PATH.exists():
    print("\nERROR: GGUF file was not downloaded.")
    sys.exit(1)

size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)

print("\nGGUF download complete:")
print(MODEL_PATH)
print(f"Size: {size_gb:.2f} GB")


# ============================================================
# 8. CREATE OLLAMA MODEL
# ============================================================

print("=" * 70)
print("8/9 - Creating Ollama model")
print("=" * 70)

MODELFILE_CONTENT = f"""FROM {MODEL_PATH}

PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

MODELFILE_PATH.write_text(
    MODELFILE_CONTENT,
    encoding="utf-8",
)

print("Modelfile:")
print(MODELFILE_CONTENT)

# Check if model already exists
status, body = http_get(f"{OLLAMA_API}/api/tags")

model_exists = (
    status == 200
    and f'"name":"{OLLAMA_MODEL}' in body
)

if model_exists:
    print(f"Model '{OLLAMA_MODEL}' already exists.")
else:
    run([
        "ollama",
        "create",
        OLLAMA_MODEL,
        "-f",
        str(MODELFILE_PATH),
    ])

print("\nInstalled Ollama models:")
run(["ollama", "list"])


# ============================================================
# 9. INSTALL + START CLOUDFLARED
# ============================================================

print("=" * 70)
print("9/9 - Installing Cloudflare Tunnel")
print("=" * 70)

if command_exists("cloudflared"):
    print("cloudflared is already installed.")
else:
    if not CLOUDFLARED_DEB.exists():
        print("Downloading cloudflared...")

        run([
            "wget",
            "-q",
            "-O",
            str(CLOUDFLARED_DEB),
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
        ])

    run([
        "dpkg",
        "-i",
        str(CLOUDFLARED_DEB),
    ], check=False)

    # Fix dependencies if necessary
    if not command_exists("cloudflared"):
        run(["apt-get", "install", "-f", "-y"])


print("\ncloudflared version:")
run(["cloudflared", "--version"])


# ============================================================
# START QUICK TUNNEL
# ============================================================

print("=" * 70)
print("Starting Cloudflare Quick Tunnel")
print("=" * 70)

# Remove previous log
try:
    CLOUDFLARED_LOG.unlink()
except FileNotFoundError:
    pass

log_file = open(CLOUDFLARED_LOG, "w")

cloudflared_process = subprocess.Popen(
    [
        "cloudflared",
        "tunnel",
        "--no-autoupdate",
        "--url",
        OLLAMA_API,
    ],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    text=True,
)

processes.append(cloudflared_process)

print("cloudflared PID:", cloudflared_process.pid)

# Wait for tunnel URL
tunnel_url = None
pattern = re.compile(
    r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com"
)

print("\nWaiting for Cloudflare URL...")

for _ in range(60):
    time.sleep(1)

    try:
        log_text = CLOUDFLARED_LOG.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        continue

    match = pattern.search(log_text)

    if match:
        tunnel_url = match.group(0)
        break

if not tunnel_url:
    print("\nERROR: Cloudflare tunnel URL was not detected.")
    print("\nCloudflared log:")
    print(CLOUDFLARED_LOG.read_text(
        encoding="utf-8",
        errors="replace",
    ))

    sys.exit(1)


# ============================================================
# TEST PUBLIC API
# ============================================================

print("\n" + "=" * 70)
print("TUNNEL READY")
print("=" * 70)

print("\nPublic Ollama URL:")
print(tunnel_url)

print("\nTesting public /api/tags endpoint...")

status, body = http_get(
    tunnel_url + "/api/tags",
    timeout=30,
)

print("HTTP status:", status)

if status == 200:
    print("PUBLIC API: OK")
else:
    print("PUBLIC API TEST FAILED")
    print(body[:2000])

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nOllama API:"
)
print(tunnel_url)

print(
    "\nModels:"
)
print(tunnel_url + "/api/tags")

print(
    "\nGenerate:"
)
print(tunnel_url + "/api/generate")

print(
    "\nChat:"
)
print(tunnel_url + "/api/chat")

print(
    "\nLocal API:"
)
print(OLLAMA_API)

print(
    "\nModel:"
)
print(OLLAMA_MODEL)

print("\nProcesses will remain running while this Kaggle session is alive.")
print("Do NOT stop this Python process if you want the tunnel to remain active.")


# ============================================================
# KEEP SCRIPT ALIVE
# ============================================================

try:
    while True:
        time.sleep(30)

        # Check Ollama
        ollama_status, _ = http_get(
            f"{OLLAMA_API}/api/tags",
            timeout=5,
        )

        # Check cloudflared
        cloudflared_alive = cloudflared_process.poll() is None

        if ollama_status != 200:
            print("WARNING: Ollama API is no longer responding.")

        if not cloudflared_alive:
            print("WARNING: cloudflared process stopped.")
            break

except KeyboardInterrupt:
    pass

finally:
    terminate_processes()
