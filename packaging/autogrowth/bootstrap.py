"""AutoGrowth desktop window and explicit browser/headless development modes."""

import argparse
import contextlib
import html
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from desktop_support import DesktopSignals, InstanceLock, user_directory
from lifecycle import Service


def configure(root):
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "AutoGrowth")]
    os.environ.pop("PYTHONHOME", None)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "AutoGrowth")]
    )
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    cache = user_directory() / "cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)


def show_error(message):
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None, message, "AutoGrowth could not start", 0x10
        )
    elif sys.platform == "darwin":
        subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                'on run argv\ndisplay alert "AutoGrowth could not start" message '
                "(item 1 of argv)\nend run",
                message,
            ],
            check=False,
        )
    else:
        print(message, file=sys.stderr)


def check():
    import growthcurves
    import matplotlib.pyplot
    import openpyxl
    import plotly
    import process_data
    import scipy.signal
    import streamlit
    import webview
    import xlrd

    import growthcurve_app.analyze

    assert (
        "Method notes file not found"
        not in growthcurve_app.analyze.load_method_notes_markdown()
    )
    print("Bundled imports and help assets OK", sys.executable)


def desktop(service, log_path):
    import webview

    stopped = threading.Event()
    exit_code = [0]
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["ALLOW_FILE_URLS"] = False
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    window = webview.create_window(
        "AutoGrowth",
        html="<h2>Starting AutoGrowth…</h2>",
        width=1280,
        height=900,
        min_size=(800, 600),
        confirm_close=True,
        localization={
            "global.quitConfirmation": "Quitting stops any running analysis. "
            "Results that you have not exported will be lost. Quit now?"
        },
    )
    window.events.closed += stopped.set

    def run():
        try:
            url = service.ready(cancelled=stopped)
            if stopped.is_set():
                return
            window.load_url(url)
            while not stopped.wait(0.25):
                if service.process.poll() is not None:
                    raise RuntimeError("The AutoGrowth server stopped unexpectedly.")
        except InterruptedError:
            pass
        except Exception as error:
            exit_code[0] = 1
            traceback.print_exc()
            if not stopped.is_set():
                window.load_html(
                    "<h2>AutoGrowth could not continue</h2><p>"
                    + html.escape(str(error))
                    + "</p><p>Log: "
                    + html.escape(str(log_path))
                    + "</p>"
                )

    try:
        # Force modern Windows renderer; never silently fall back to IE or a browser.
        with DesktopSignals(service.close):
            webview.start(
                run,
                gui="edgechromium" if sys.platform == "win32" else "cocoa",
                private_mode=True,
            )
    finally:
        stopped.set()
    return exit_code[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Explicit external-browser mode; Ctrl+C stops the server",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Headless lifecycle test mode; Ctrl+C stops the server",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    configure(root)
    if args.check:
        check()
        return 0
    logs = user_directory() / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f'autogrowth-{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}.log'
    try:
        with (
            InstanceLock(),
            tempfile.TemporaryDirectory(prefix="autogrowth-session-") as state,
        ):
            with log_path.open("a", encoding="utf-8") as log:
                headless = args.browser or args.no_browser
                with contextlib.ExitStack() as stack:
                    if not headless:
                        stack.enter_context(contextlib.redirect_stdout(log))
                        stack.enter_context(contextlib.redirect_stderr(log))
                    service = Service(root, state, log_path)
                    try:
                        if headless:

                            def interrupted(*_):
                                raise KeyboardInterrupt

                            signal.signal(signal.SIGTERM, interrupted)
                            signal.signal(signal.SIGINT, interrupted)
                            url = service.ready()
                            print(
                                f"AutoGrowth: {url}\nLog: {log_path}\nCtrl+C stops the "
                                "server; closing a browser tab does not.",
                                flush=True,
                            )
                            if args.browser:
                                webbrowser.open(url)
                            return service.process.wait()
                        return desktop(service, log_path)
                    except KeyboardInterrupt:
                        return 0
                    finally:
                        service.close()
    except Exception as error:
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        message = f"{error}\n\nLog: {log_path}"
        if args.browser or args.no_browser:
            print(message, file=sys.stderr)
        else:
            show_error(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
