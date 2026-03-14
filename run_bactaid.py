#!/usr/bin/env python3
"""
BactAID Windows Launcher

This script:
1. Checks if Ollama is running
2. Starts the Flask backend server
3. Opens the frontend in the default browser
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
from pathlib import Path

# Configuration
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Colors for console output (Windows compatible)
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def enable_windows_colors():
    """Enable ANSI color codes on Windows."""
    if sys.platform == "win32":
        os.system("")  # Enables ANSI escape sequences

def print_banner():
    """Print the BactAID banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  █████╗  ██████╗████████╗ █████╗ ██╗██████╗        ║
║   ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║██╔══██╗       ║
║   ██████╔╝███████║██║        ██║   ███████║██║██║  ██║       ║
║   ██╔══██╗██╔══██║██║        ██║   ██╔══██║██║██║  ██║       ║
║   ██████╔╝██║  ██║╚██████╗   ██║   ██║  ██║██║██████╔╝       ║
║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═════╝        ║
║                                                              ║
║        AI-Powered Bacterial Identification System            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
    print(banner)

def is_port_open(host: str, port: int) -> bool:
    """Check if a port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_ollama() -> bool:
    """Check if Ollama is running."""
    print(f"{Colors.CYAN}[1/4] Checking Ollama status...{Colors.RESET}")

    if is_port_open(OLLAMA_HOST, OLLAMA_PORT):
        print(f"  {Colors.GREEN}✓ Ollama is running on port {OLLAMA_PORT}{Colors.RESET}")
        return True
    else:
        print(f"  {Colors.YELLOW}⚠ Ollama is not running{Colors.RESET}")
        print(f"  {Colors.YELLOW}  Please start Ollama before using BactAID{Colors.RESET}")
        print(f"  {Colors.YELLOW}  Run: ollama serve{Colors.RESET}")
        return False

def check_ollama_model() -> bool:
    """Check if the required Ollama model is available."""
    print(f"{Colors.CYAN}[2/4] Checking Ollama model...{Colors.RESET}")

    try:
        import ollama
        client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
        models = client.list()
        model_names = [m.get("name", "") for m in models.get("models", [])]

        # Check for llama3.2:3b or similar
        target_model = os.getenv("BACTAI_OLLAMA_MODEL", "llama3.2:3b")
        has_model = any(target_model in name or name.startswith("llama3.2") for name in model_names)

        if has_model:
            print(f"  {Colors.GREEN}✓ Model '{target_model}' is available{Colors.RESET}")
            return True
        else:
            print(f"  {Colors.YELLOW}⚠ Model '{target_model}' not found{Colors.RESET}")
            print(f"  {Colors.YELLOW}  Run: ollama pull {target_model}{Colors.RESET}")
            if model_names:
                print(f"  {Colors.YELLOW}  Available models: {', '.join(model_names[:5])}{Colors.RESET}")
            return False
    except ImportError:
        print(f"  {Colors.YELLOW}⚠ Cannot check model (ollama package not installed){Colors.RESET}")
        return True  # Continue anyway
    except Exception as e:
        print(f"  {Colors.YELLOW}⚠ Cannot check model: {e}{Colors.RESET}")
        return True  # Continue anyway

def check_backend_not_running() -> bool:
    """Check that backend port is free."""
    print(f"{Colors.CYAN}[3/4] Checking backend port...{Colors.RESET}")

    if is_port_open(BACKEND_HOST, BACKEND_PORT):
        print(f"  {Colors.YELLOW}⚠ Port {BACKEND_PORT} is already in use{Colors.RESET}")
        print(f"  {Colors.YELLOW}  Another instance may be running{Colors.RESET}")
        return False
    else:
        print(f"  {Colors.GREEN}✓ Port {BACKEND_PORT} is available{Colors.RESET}")
        return True

def start_backend():
    """Start the Flask backend server."""
    print(f"{Colors.CYAN}[4/4] Starting backend server...{Colors.RESET}")

    # Determine the project root
    script_dir = Path(__file__).parent.absolute()
    backend_script = script_dir / "backend" / "app.py"

    if not backend_script.exists():
        print(f"  {Colors.RED}✗ Backend script not found: {backend_script}{Colors.RESET}")
        return None

    # Start the backend in a subprocess
    try:
        if sys.platform == "win32":
            # On Windows, use CREATE_NEW_PROCESS_GROUP to allow clean shutdown
            process = subprocess.Popen(
                [sys.executable, str(backend_script)],
                cwd=str(script_dir),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            process = subprocess.Popen(
                [sys.executable, str(backend_script)],
                cwd=str(script_dir),
            )

        print(f"  {Colors.GREEN}✓ Backend starting (PID: {process.pid})...{Colors.RESET}")
        return process
    except Exception as e:
        print(f"  {Colors.RED}✗ Failed to start backend: {e}{Colors.RESET}")
        return None

def wait_for_backend(timeout: int = 30) -> bool:
    """Wait for backend to become available."""
    print(f"  Waiting for backend to be ready...", end="", flush=True)

    for i in range(timeout):
        if is_port_open(BACKEND_HOST, BACKEND_PORT):
            print(f" {Colors.GREEN}ready!{Colors.RESET}")
            return True
        print(".", end="", flush=True)
        time.sleep(1)

    print(f" {Colors.RED}timeout{Colors.RESET}")
    return False

def open_browser():
    """Open the frontend in the default browser."""
    print(f"\n{Colors.CYAN}Opening browser...{Colors.RESET}")

    # Check if frontend/dist/index.html exists for static serving
    script_dir = Path(__file__).parent.absolute()
    frontend_dist = script_dir / "frontend" / "dist" / "index.html"

    if frontend_dist.exists():
        # Serve via backend static files
        url = FRONTEND_URL
    else:
        # Development mode - assume Vite dev server
        url = "http://localhost:3000"

    print(f"  {Colors.GREEN}Opening: {url}{Colors.RESET}")
    webbrowser.open(url)

def main():
    """Main entry point."""
    enable_windows_colors()
    print_banner()

    print(f"{Colors.BOLD}Starting BactAID...{Colors.RESET}\n")

    # Step 1: Check Ollama
    ollama_running = check_ollama()

    # Step 2: Check model (only if Ollama is running)
    if ollama_running:
        check_ollama_model()

    # Step 3: Check port
    if not check_backend_not_running():
        response = input(f"\n{Colors.YELLOW}Continue anyway? (y/N): {Colors.RESET}").strip().lower()
        if response != 'y':
            print(f"{Colors.RED}Aborted.{Colors.RESET}")
            sys.exit(1)

    # Step 4: Start backend
    backend_process = start_backend()
    if backend_process is None:
        print(f"\n{Colors.RED}Failed to start backend. Exiting.{Colors.RESET}")
        sys.exit(1)

    # Wait for backend to be ready
    if not wait_for_backend():
        print(f"\n{Colors.RED}Backend failed to start. Check the logs above.{Colors.RESET}")
        backend_process.terminate()
        sys.exit(1)

    # Open browser
    open_browser()

    # Print status
    print(f"\n{Colors.GREEN}{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"{Colors.GREEN}BactAID is running!{Colors.RESET}")
    print(f"{Colors.GREEN}═══════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"\n  Backend: {FRONTEND_URL}")
    print(f"  Ollama:  {'Connected' if ollama_running else 'Not connected (RAG will use templates)'}")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.RESET}\n")

    # Wait for the backend process
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}Shutting down...{Colors.RESET}")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
        print(f"{Colors.GREEN}Goodbye!{Colors.RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
