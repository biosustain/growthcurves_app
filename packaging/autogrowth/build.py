"""Build a single-runtime portable AutoGrowth folder on the matching target host."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON_VERSION = "3.12.11"
WINDOWS_PYTHON_VERSION = "3.12.10"  # Last official 3.12 Windows embeddable release.
EXAMPLES = (
    "batch_example/example_batch_data_od_readings.csv",
    "batch_example/example_batch_data_od_readings_calibration.csv",
    "turbidostat_example/example_2-Pio_Experiment_dilution_events.csv",
)


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def trim_runtime(runtime):
    # Explicit package allowlist: do not remove arbitrary folders named tests.
    site = runtime / (
        "Lib/site-packages"
        if platform.system() == "Windows"
        else "lib/python3.12/site-packages"
    )
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "pyarrow",
        "numba",
        "llvmlite",
    ):
        for tests in sorted(
            (site / package).rglob("tests"), key=lambda p: len(p.parts), reverse=True
        ):
            if tests.is_dir():
                shutil.rmtree(tests)
    for path in list(runtime.rglob("__pycache__")):
        shutil.rmtree(path)
    for path in runtime.rglob("*.a"):
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; never overwritten",
    )
    parser.add_argument("--uv", default="uv", help="Build-time uv executable")
    parser.add_argument("--refresh-lock", action="store_true")
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Build a macOS .app or Windows native launcher",
    )
    parser.add_argument(
        "--dmg",
        action="store_true",
        help="Also create a macOS DMG (requires --desktop)",
    )
    parser.add_argument(
        "--installer",
        action="store_true",
        help="Also compile Windows Inno Setup installer",
    )
    parser.add_argument("--webview2-bootstrapper", type=Path)
    parser.add_argument("--iscc", default="ISCC.exe")
    args = parser.parse_args()
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "x86_64"):
        target = f"macos-{machine}"
    elif system == "Windows" and machine in ("amd64", "x86_64"):
        target = "windows-x64"
    else:
        parser.error("Build on macOS or Windows x64; cross-building is not supported.")
    if args.dmg and (system != "Darwin" or not args.desktop):
        parser.error("--dmg requires --desktop on macOS")
    if args.installer and (
        system != "Windows" or not args.desktop or not args.webview2_bootstrapper
    ):
        parser.error(
            "--installer requires --desktop and --webview2-bootstrapper on Windows"
        )
    if args.desktop and system == "Darwin" and args.output.suffix != ".app":
        parser.error("macOS desktop output must end in .app")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if system == "Windows" and not shutil.which("pwsh"):
        parser.error("PowerShell 7 (pwsh) is required to verify the Windows runtime.")
    if args.desktop:
        native_tools = (
            ["clang", "sips", "iconutil"] if system == "Darwin" else ["cl", "rc"]
        )
        if args.dmg:
            native_tools.append("hdiutil")
            if output.with_suffix(".dmg").exists():
                parser.error("DMG output already exists")
        if args.installer:
            native_tools.append(args.iscc)
            if not args.webview2_bootstrapper.is_file():
                parser.error("WebView2 bootstrapper does not exist")
        for tool in native_tools:
            if not shutil.which(tool):
                parser.error(f"Required build tool not on PATH: {tool}")
    # Preflight every input before downloading dependencies.
    required = [
        ROOT / "AutoGrowth/app.py",
        ROOT / "AutoGrowth/logo.svg",
        ROOT / "src/growthcurve_app/__init__.py",
        ROOT / "LICENSE",
    ]
    required += [ROOT / "AutoGrowth/data" / path for path in EXAMPLES]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = HERE / f"requirements-{target}.lock"
    if not lock.exists() and not args.refresh_lock:
        parser.error(
            f"No lock for {target}; first run with --refresh-lock and review/commit the lock."
        )
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["UV_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "autogrowth-uv-cache")
    env["UV_LINK_MODE"] = "copy"
    with tempfile.TemporaryDirectory(prefix="autogrowth-build-") as temp:
        staging = Path(temp)
        bundle = staging / "AutoGrowth"
        runtime = bundle / "runtime"
        python_version = WINDOWS_PYTHON_VERSION if system == "Windows" else PYTHON_VERSION
        if system == "Windows":
            run(
                "pwsh", "-NoProfile", "-NonInteractive", "-File",
                HERE / "windows_runtime.ps1",
                "-Destination", runtime, "-Version", python_version,
            )
        else:
            install = staging / "python-download"
            run(
                args.uv,
                "python",
                "install",
                python_version,
                "--install-dir",
                install,
                "--no-bin",
                "--no-registry",
                env=env,
            )
            runtimes = list(install.glob("cpython-*"))
            if len(runtimes) != 1:
                raise RuntimeError(f"Expected exactly one Python runtime, found {runtimes}")
            shutil.copytree(runtimes[0], runtime, symlinks=True)
        python = runtime / ("python.exe" if system == "Windows" else "bin/python3")
        if args.refresh_lock:
            run(
                args.uv,
                "pip",
                "compile",
                HERE / "requirements.in",
                "--python",
                python,
                "--generate-hashes",
                "--output-file",
                lock,
                env=env,
                stdout=subprocess.DEVNULL,
            )
        run(
            args.uv,
            "pip",
            "install",
            "--python",
            python,
            "--require-hashes",
            "--break-system-packages",
            *[
                arg
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "matplotlib",
                    "pyarrow",
                    "llvmlite",
                    "numba",
                    "scikit-learn",
                    "pillow",
                )
                for arg in ("--only-binary", name)
            ],
            "-r",
            lock,
            env=env,
        )
        app = bundle / "AutoGrowth"
        app.mkdir()
        for path in (ROOT / "AutoGrowth").glob("*.py"):
            shutil.copy2(path, app / path.name)
        for name in ("VERSION", "logo.svg"):
            shutil.copy2(ROOT / "AutoGrowth" / name, app / name)
        shutil.copytree(ROOT / "AutoGrowth/markdowns", app / "markdowns")
        (bundle / ".streamlit").mkdir()
        shutil.copy2(
            ROOT / "AutoGrowth/.streamlit/config.toml",
            bundle / ".streamlit/config.toml",
        )
        for relative in EXAMPLES:
            destination = app / "data" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "AutoGrowth/data" / relative, destination)
        shutil.copytree(
            ROOT / "src/growthcurve_app",
            bundle / "src/growthcurve_app",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        # Source imports use distribution metadata; no editable project installation.
        version = (app / "VERSION").read_text().strip()
        metadata = bundle / "src" / f"growthcurve_app-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: growthcurve-app\nVersion: {version}\n"
        )
        (bundle / "info_plots").mkdir()
        for path in (ROOT / "info_plots").glob("*.png"):
            shutil.copy2(path, bundle / "info_plots" / path.name)
        for source, name in [
            (ROOT / "LICENSE", "LICENSE"),
            (HERE / "bootstrap.py", "bootstrap.py"),
            (lock, "requirements.lock"),
            (HERE / "README.md", "BUILDING.md"),
        ]:
            shutil.copy2(source, bundle / name)
        for name in ("desktop_support.py", "lifecycle.py", "server_runner.py"):
            shutil.copy2(HERE / name, bundle / name)
        if system == "Windows":
            (bundle / "AutoGrowth.cmd").write_text(
                '@echo off\r\nset "PYTHONHOME="\r\nset "PYTHONPATH="\r\n"%~dp0runtime\\python.exe" -s -B "%~dp0bootstrap.py" --browser %*\r\nexit /b %errorlevel%\r\n'
            )
        else:
            launcher = bundle / "AutoGrowth.command"
            launcher.write_text(
                '#!/bin/sh\nunset PYTHONHOME PYTHONPATH\nROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec "$ROOT/runtime/bin/python3" -s -B "$ROOT/bootstrap.py" --browser "$@"\n'
            )
            launcher.chmod(0o755)
        trim_runtime(runtime)
        for path in runtime.rglob("*"):
            if path.is_symlink() and not path.resolve().is_relative_to(
                runtime.resolve()
            ):
                raise RuntimeError(f"Runtime contains external symlink: {path}")
        # Test after relocation to a path with spaces, with developer tools absent from PATH.
        moved = staging / "Relocated AutoGrowth"
        bundle.rename(moved)
        python = moved / python.relative_to(bundle)
        check_env = env.copy()
        check_env["PATH"] = (
            os.environ.get("SystemRoot", "C:\\Windows") + "\\System32"
            if system == "Windows"
            else "/usr/bin:/bin"
        )
        check_env["PYTHONNOUSERSITE"] = "1"
        run(
            python,
            "-s",
            "-B",
            moved / "bootstrap.py",
            "--check",
            cwd=staging,
            env=check_env,
            timeout=180,
        )
        run(
            python,
            "-s",
            "-B",
            HERE / "verify.py",
            moved,
            cwd=staging,
            env=check_env,
            timeout=240,
        )
        if args.desktop:
            run(
                python,
                "-s",
                "-B",
                HERE / "test_lifecycle.py",
                moved,
                cwd=staging,
                env=check_env,
                timeout=300,
            )
        trim_runtime(moved / "runtime")
        if system == "Windows":
            run(
                "pwsh", "-NoProfile", "-NonInteractive", "-File",
                HERE / "windows_runtime.ps1",
                "-Destination", moved / "runtime", "-Version", python_version,
                "-VerifyOnly",
            )
        sizes = {}
        for path in moved.rglob("*"):
            if path.is_file():
                group = path.relative_to(moved).parts[0]
                sizes[group] = sizes.get(group, 0) + path.stat().st_size
        (moved / "build-report.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "python": python_version,
                    "sizes_bytes": sizes,
                    "validation": "relocated imports, page renders, representative fit, and Excel round-trip",
                },
                indent=2,
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.desktop:
            from native_package import (
                make_dmg,
                make_installer,
                wrap_macos,
                wrap_windows,
            )

            if system == "Darwin":
                wrap_macos(moved, output)
                if args.dmg:
                    make_dmg(output, output.with_suffix(".dmg"))
            else:
                wrap_windows(moved)
                shutil.move(str(moved), str(output))
                if args.installer:
                    make_installer(
                        output, output.parent, args.webview2_bootstrapper, args.iscc
                    )
        else:
            shutil.move(str(moved), str(output))
        print(f"Built {output}: {sum(sizes.values()) / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
