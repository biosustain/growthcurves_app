"""Real subprocess tests. Run with the bundled Python and payload path as argv[1]."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(sys.argv.pop(1)).resolve() if __name__ == '__main__' else None
if ROOT is not None:
    sys.path.insert(0, str(ROOT))
    from lifecycle import Service


def wait_for(predicate, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.1)
    raise AssertionError('Timed out waiting for process cleanup')


def alive(pid):
    if sys.platform == 'win32':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        k.OpenProcess.restype = ctypes.c_void_p
        k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return k.WaitForSingleObject(handle, 0) == 258
        finally:
            k.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def port_closed(port):
    with socket.socket() as sock:
        sock.settimeout(.2)
        return sock.connect_ex(('127.0.0.1', port)) != 0


@unittest.skipIf(ROOT is None, 'Run this integration script with a bundled payload path')
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='AutoGrowth lifecycle ')
        self.path = Path(self.temp.name)
        self.log = self.path / 'server.log'
        self.service = None

    def tearDown(self):
        if self.service:
            self.service.close()
        self.temp.cleanup()

    def start(self):
        self.service = Service(ROOT, self.path, self.log)
        self.service.ready()
        return json.loads(self.service.status.read_text())

    def assert_stopped(self, info):
        wait_for(lambda: port_closed(info['port']) and not alive(info['pid']))

    def test_graceful_close(self):
        info = self.start()
        self.service.close()
        self.assert_stopped(info)
        self.assertIn('AUTOGROWTH_GRACEFUL_STOP', self.log.read_text())

    def test_close_during_startup(self):
        self.service = Service(ROOT, self.path, self.log)
        self.service.close()
        self.assertIsNotNone(self.service.process.returncode)
        if self.service.status.exists():
            self.assert_stopped(json.loads(self.service.status.read_text()))

    def test_supervisor_killed(self):
        info = self.start()
        self.service.process.kill()
        self.service.process.wait(timeout=5)
        self.assert_stopped(info)
        if sys.platform != 'win32':
            self.assertIn('AUTOGROWTH_GRACEFUL_STOP', self.log.read_text())

    @unittest.skipIf(sys.platform == 'win32', 'SIGSTOP fault injection is POSIX-only')
    def test_stalled_server_is_killed(self):
        info = self.start()
        os.kill(info['pid'], signal.SIGSTOP)
        try:
            self.service.close()
            self.assert_stopped(info)
        finally:
            if alive(info['pid']):
                os.kill(info['pid'], signal.SIGKILL)

    def test_gui_owner_killed(self):
        # Same pipe ownership as bootstrap, without needing a GUI in this test.
        driver = self.path / 'owner.py'
        driver.write_text('import sys, time\nfrom pathlib import Path\n'
            f'sys.path.insert(0, {str(ROOT)!r})\nfrom lifecycle import Service\n'
            f's = Service(Path({str(ROOT)!r}), Path({str(self.path)!r}), Path({str(self.log)!r}))\n'
            's.ready()\ntime.sleep(120)\n')
        owner = subprocess.Popen([sys.executable, '-s', '-B', str(driver)])
        info = None
        try:
            wait_for(lambda: (self.path / 'server.json').exists(), timeout=60)
            info = json.loads((self.path / 'server.json').read_text())
            owner.kill()
            owner.wait(timeout=5)
            self.assert_stopped(info)
            self.assertIn('AUTOGROWTH_GRACEFUL_STOP', self.log.read_text())
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()
            if info and alive(info['pid']):
                os.kill(info['pid'], signal.SIGTERM)

    def test_startup_failure(self):
        fixture = self.path / 'broken payload'
        fixture.mkdir()
        for name in ('lifecycle.py', 'desktop_support.py'):
            shutil.copy2(ROOT / name, fixture / name)
        (fixture / 'server_runner.py').write_text('raise RuntimeError("intentional startup failure")\n')
        self.service = Service(fixture, self.path, self.log)
        with self.assertRaises(RuntimeError):
            self.service.ready(timeout=10)
        self.service.close()
        self.assertIn('intentional startup failure', self.log.read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
