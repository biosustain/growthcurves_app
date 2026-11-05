"""Private pipe supervisor: one Python installation, independently owned processes."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from desktop_support import WindowsJob, InstanceLock

SHUTDOWN_TIMEOUT = 12


def process_options():
    if sys.platform == 'win32':
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {'start_new_session': True}


def python_executable():
    path = Path(sys.executable)
    return str(path.with_name('python.exe') if path.name.lower() == 'pythonw.exe' else path)


def kill_tree(process):
    if sys.platform == 'win32':
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_child(child):
    if child.stdin:
        child.stdin.close()  # EOF is the graceful shutdown request, even without a console.
    try:
        child.wait(timeout=SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        kill_tree(child)
        child.wait(timeout=5)
    finally:
        # Clean remaining descendants even if the main server has already exited.
        if sys.platform != 'win32':
            kill_tree(child)


def supervise(root, status):
    stop = threading.Event()
    def read_parent():
        sys.stdin.buffer.readline()
        stop.set()
    threading.Thread(target=read_parent, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mutex = InstanceLock(allow_existing=True) if sys.platform == "win32" else None
    if mutex:
        mutex.__enter__()
    job = WindowsJob()
    child = None
    try:
        child = subprocess.Popen([python_executable(), '-s', '-B', str(root / 'server_runner.py'),
                                  str(port), str(status)], stdin=subprocess.PIPE, cwd=root,
                                 **process_options())
        job.assign(child)
        child.stdin.write(b'START\n')
        child.stdin.flush()
        while not stop.wait(0.1):
            if child.poll() is not None:
                return child.returncode or 1  # Unexpected server exit is an error.
        return 0
    finally:
        try:
            if child is not None:
                stop_child(child)
        finally:
            job.close()
            if mutex:
                mutex.__exit__()


class Service:
    def __init__(self, root, state_dir, log_path):
        self._close_lock = threading.Lock()
        self.root = Path(root)
        self.status = Path(state_dir) / 'server.json'
        self.status.unlink(missing_ok=True)
        self.log = Path(log_path).open('a', encoding='utf-8')
        try:
            self.process = subprocess.Popen(
                [python_executable(), '-s', '-B', str(self.root / 'lifecycle.py'), str(self.root), str(self.status)],
                stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
                cwd=self.root, **process_options())
        except BaseException:
            self.log.close()
            raise

    def ready(self, timeout=60, cancelled=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancelled and cancelled.is_set():
                raise InterruptedError('Startup cancelled')
            if self.process.poll() is not None:
                raise RuntimeError('AutoGrowth server failed to start. See the log for details.')
            if self.status.exists():
                info = json.loads(self.status.read_text(encoding='utf-8'))
                url = f'http://127.0.0.1:{info["port"]}'
                try:
                    with urllib.request.urlopen(url + '/_stcore/health', timeout=0.5) as response:
                        if response.status == 200:
                            return url
                except OSError:
                    pass
            time.sleep(0.1)
        raise TimeoutError('AutoGrowth did not become ready within 60 seconds.')

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=SHUTDOWN_TIMEOUT + 6)
            except subprocess.TimeoutExpired:
                kill_tree(self.process)
                self.process.wait(timeout=5)
                # Server pipe EOF triggers its own 15-second watchdog if supervisor was stuck.
        finally:
            self.log.close()


if __name__ == '__main__':
    raise SystemExit(supervise(Path(sys.argv[1]), Path(sys.argv[2])))
