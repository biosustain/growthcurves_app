"""Pinned Streamlit adapter with a private stdin shutdown channel.

No public HTTP shutdown endpoint. EOF also requests shutdown if the supervisor dies.
"""

import asyncio
import json
import os
import signal
import sys
import threading
from pathlib import Path


def main():
    port, status_path = int(sys.argv[1]), Path(sys.argv[2])
    # The supervisor assigns the Windows Job Object before allowing any app code.
    if sys.stdin.buffer.readline() != b"START\n":
        return
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "AutoGrowth")]
    stopped = threading.Event()
    requested = threading.Event()

    def request_stop():
        if requested.is_set():
            return
        requested.set()

        def hard_deadline():
            if not stopped.wait(15):
                if sys.platform != "win32":
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                os._exit(2)

        threading.Thread(target=hard_deadline, daemon=True).start()

    def monitor():
        sys.stdin.buffer.readline()  # STOP or EOF; neither is exposed to the browser.
        request_stop()

    threading.Thread(target=monitor, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: request_stop())
    signal.signal(signal.SIGINT, lambda *_: request_stop())
    import streamlit

    if streamlit.__version__ != "1.66.1":
        raise RuntimeError(
            "Desktop server adapter requires Streamlit 1.66.1; rerun lifecycle tests before upgrading."
        )
    from streamlit import config
    from streamlit.web.bootstrap import load_config_options
    from streamlit.web.server import Server

    load_config_options(
        {
            "server_address": "127.0.0.1",
            "server_port": port,
            "server_headless": True,
            "server_fileWatcherType": "none",
            "browser_gatherUsageStats": False,
        }
    )
    config._server_mode = "starlette-managed"
    script = str(root / "AutoGrowth/app.py")
    sys.argv = [script]

    async def serve():
        if requested.is_set():
            return
        server = Server(script, False)
        try:
            await server.start()
            temp = status_path.with_suffix(".tmp")
            temp.write_text(
                json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8"
            )
            temp.replace(status_path)
            while not requested.is_set():
                await asyncio.sleep(0.1)
        finally:
            server.stop()
            await asyncio.wait_for(server.stopped, timeout=10)

    try:
        asyncio.run(serve())
        print("AUTOGROWTH_GRACEFUL_STOP", flush=True)
    finally:
        stopped.set()


if __name__ == "__main__":
    main()
