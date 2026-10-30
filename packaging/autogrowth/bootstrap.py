"""Run AutoGrowth using only its bundled interpreter and the system browser."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "AutoGrowth")]
    if args.check:
        import growthcurves
        import matplotlib.pyplot
        import openpyxl
        import plotly
        import process_data
        import scipy.signal
        import streamlit
        import xlrd

        import growthcurve_app.analyze

        assert (
            "Method notes file not found"
            not in growthcurve_app.analyze.load_method_notes_markdown()
        )
        print("Bundled imports and help assets OK", sys.executable)
        return 0
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "AutoGrowth")])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_dir = Path(tempfile.gettempdir()) / "AutoGrowth-logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"server-{os.getpid()}.log"
    print(f"AutoGrowth log: {log_path}", flush=True)
    command = [
        sys.executable,
        "-s",
        "-B",
        "-m",
        "streamlit",
        "run",
        str(root / "AutoGrowth/app.py"),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(
                        f"Server exited ({child.returncode}); see {log_path}"
                    )
                try:
                    with urllib.request.urlopen(
                        url + "/_stcore/health", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"Server startup timed out; see {log_path}")
            if child.poll() is not None:
                raise RuntimeError(f"Server exited; see {log_path}")
            print(
                f"AutoGrowth: {url}\nPress Ctrl+C here to stop."
                " Closing the browser tab does not stop the server.",
                flush=True,
            )
            if not args.no_browser and not webbrowser.open(url):
                print("Open the URL above in your browser.", flush=True)
            return child.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
