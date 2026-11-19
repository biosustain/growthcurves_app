"""Per-user paths, single-instance ownership, and Windows process containment."""

import ctypes
import os
import sys
from pathlib import Path

MUTEX_NAME = "Local\\AutoGrowth.Desktop"


def user_directory():
    if sys.platform == "win32":
        root = Path(os.environ["LOCALAPPDATA"]) / "AutoGrowth"
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support/AutoGrowth"
    else:
        root = Path.home() / ".local/share/AutoGrowth"
    root.mkdir(parents=True, exist_ok=True)
    return root


class InstanceLock:
    def __init__(self, allow_existing=False):
        self.allow_existing = allow_existing

    def __enter__(self):
        if sys.platform == "win32":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_wchar_p,
            ]
            kernel.CreateMutexW.restype = ctypes.c_void_p
            self.handle = kernel.CreateMutexW(None, False, MUTEX_NAME)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if ctypes.get_last_error() == 183 and not self.allow_existing:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle(self.handle)
                raise RuntimeError(
                    "AutoGrowth is already running. Close its existing window first."
                )
            self.kernel = kernel
        else:
            import fcntl

            self.file = (user_directory() / "desktop.lock").open("a")
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.file.close()
                raise RuntimeError(
                    "AutoGrowth is already running. Close its existing window first."
                ) from None
        return self

    def __exit__(self, *_):
        if sys.platform == "win32":
            self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            self.kernel.CloseHandle(self.handle)
        else:
            self.file.close()


class WindowsJob:
    """Supervisor owns a non-inheritable kill-on-close Job Object."""

    def __init__(self):
        self.handle = None
        if sys.platform != "win32":
            return
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", w.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", w.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", w.DWORD),
                ("SchedulingClass", w.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            w.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            w.DWORD,
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.BasicLimitInformation.LimitFlags = (
            0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if self.handle and not self.kernel.AssignProcessToJobObject(
            self.handle, int(process._handle)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class DesktopSignals:
    """Signal wakeup fd works while Cocoa keeps Python's main thread in native code.

    Only explicit OS termination uses this path. Window close/Cancel use GUI events.
    """

    def __init__(self, cleanup):
        self.cleanup = cleanup

    def __enter__(self):
        import signal
        import socket
        import threading

        self.reader, self.writer = socket.socketpair()
        self.writer.setblocking(False)
        self.previous_fd = signal.set_wakeup_fd(self.writer.fileno())
        self.handlers = {}
        for number in (signal.SIGTERM, signal.SIGINT):
            self.handlers[number] = signal.signal(number, lambda *_: None)

        def receive():
            try:
                while True:
                    data = self.reader.recv(1)
                    if not data or data[0] in (signal.SIGTERM, signal.SIGINT):
                        break
            except OSError:
                return
            if data:
                try:
                    self.cleanup()
                finally:
                    # Cocoa may still be inside a modal/native event loop. All owned
                    # server processes have been stopped (or their watchdog armed).
                    for stream in (sys.stdout, sys.stderr):
                        if stream:
                            try:
                                stream.flush()
                            except (OSError, ValueError):
                                pass
                    os._exit(0)

        threading.Thread(target=receive, daemon=True).start()
        return self

    def __exit__(self, *_):
        import signal

        signal.set_wakeup_fd(self.previous_fd)
        for number, handler in self.handlers.items():
            signal.signal(number, handler)
        self.writer.close()
        self.reader.close()
