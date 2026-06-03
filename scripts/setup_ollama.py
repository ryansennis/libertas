#!/usr/bin/env python3
"""
Install Ollama, pull a model, and serve it on a local port.

Usage:
    python ollama_setup.py
    python ollama_setup.py --model mistral
    python ollama_setup.py --model tinyllama --port 11435
"""

import argparse
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time


OLLAMA_LINUX_INSTALL_URL = "https://ollama.com/install.sh"
DEFAULT_MODEL = "tinyllama"
DEFAULT_PORT = 11434


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


# ── Installation ──────────────────────────────────────────────────────────────

def is_installed() -> bool:
    return shutil.which("ollama") is not None


def install_ollama():
    system = platform.system()
    print(f"\n[1/3] Installing Ollama on {system}...")

    if system == "Darwin":
        if shutil.which("brew") is None:
            sys.exit(
                "Homebrew not found. Install it from https://brew.sh or install "
                "Ollama manually from https://ollama.com/download"
            )
        run(["brew", "install", "ollama"])

    elif system == "Linux":
        run(
            ["sh", "-c", f"curl -fsSL {OLLAMA_LINUX_INSTALL_URL} | sh"],
            shell=False,
        )

    elif system == "Windows":
        sys.exit(
            "Automated Windows install isn't supported here.\n"
            "Download the installer from https://ollama.com/download/windows"
        )

    else:
        sys.exit(f"Unsupported OS: {system}")

    if not is_installed():
        sys.exit("Installation succeeded but 'ollama' not found on PATH. "
                 "Try opening a new terminal.")

    print("  ✓ Ollama installed.")


# ── Server helpers ────────────────────────────────────────────────────────────

def start_server_background(port: int) -> subprocess.Popen:
    """Start ollama serve in the background and wait until it accepts connections."""
    env = {**os.environ, "OLLAMA_HOST": f"0.0.0.0:{port}"}
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Poll until the port is open (up to 15s)
    for _ in range(30):
        time.sleep(0.5)
        if proc.poll() is not None:
            sys.exit("Ollama server exited unexpectedly. Check for port conflicts.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc
        except OSError:
            pass
    sys.exit(f"Ollama server did not start within 15s on port {port}.")


# ── Model pull ────────────────────────────────────────────────────────────────

def pull_model(model: str, port: int):
    print(f"\n[2/3] Pulling model '{model}' (this may take a while)...")
    env = {**os.environ, "OLLAMA_HOST": f"0.0.0.0:{port}"}
    run(["ollama", "pull", model], env=env)
    print(f"  ✓ Model '{model}' ready.")


# ── Serve ─────────────────────────────────────────────────────────────────────

def serve(model: str, port: int, proc: subprocess.Popen):
    """Hand off an already-running background server process to the foreground."""
    print(f"  ✓ Server running (PID {proc.pid})")
    print(f"  ✓ API base: http://localhost:{port}")
    print(f"  ✓ Model:    {model}")
    print("\nPress Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        print("\nShutting down...")
        proc.terminate()
        proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    proc.wait()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Ollama, pull a model, and serve it.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to pull (default: {DEFAULT_MODEL})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to serve on (default: {DEFAULT_PORT})")
    parser.add_argument("--skip-install", default=False, action="store_true", help="Skip installation even if ollama isn't found")
    parser.add_argument("--skip-pull", default=False, action="store_true", help="Skip model pull (use if already pulled)")
    parser.add_argument("--skip-serve", default=False, action="store_true", help="Skip serving the model")
    parser.add_argument("--daemon", action="store_true", help="Run server in foreground (blocks terminal until Ctrl+C)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_install:
        if is_installed():
            print("[1/3] Ollama already installed — skipping.")
        else:
            install_ollama()
    else:
        print("[1/3] Skipping install (--skip-install).")

    # Server must be running before we can pull
    if not args.skip_serve:
        print(f"\n[2/3] Starting Ollama server on http://localhost:{args.port} ...")
        server_proc = start_server_background(args.port)
        print("  ✓ Server ready.")
    else:
        print("[2/3] Skipping serve (--skip-serve).")
        server_proc = None

    if not args.skip_pull:
        pull_model(args.model, args.port)
    else:
        print(f"[3/3] Skipping pull (--skip-pull).")

    if not args.skip_serve and server_proc:
        print(f"\n[3/3] Server started in background.")
        print(f"  ✓ PID:      {server_proc.pid}")
        print(f"  ✓ API base: http://localhost:{args.port}")
        print(f"  ✓ Model:    {args.model}")
        print(f"\nTo stop the server: kill {server_proc.pid}")
        print(f"Or use: ollama stop")

        if args.daemon:
            print("\nRunning in daemon mode. Press Ctrl+C to stop.\n")
            serve(args.model, args.port, server_proc)
        else:
            print("\nServer running in background. Terminal is available for use.")


if __name__ == "__main__":
    main()