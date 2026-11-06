"""Wrap the existing single-runtime payload; do not freeze or duplicate Python."""

import os
import plistlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def wrap_macos(payload, output):
    output = Path(output)
    if output.suffix != ".app":
        raise ValueError("macOS desktop output must end in .app")
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="autogrowth-native-") as temp:
        temp = Path(temp)
        app = temp / "AutoGrowth.app"
        contents = app / "Contents"
        resources = contents / "Resources"
        binaries = contents / "MacOS"
        resources.mkdir(parents=True)
        binaries.mkdir()
        # Match the installed wheel platform floors rather than the build host OS.
        minimums = [(12, 0)]
        for wheel in Path(payload).glob(
            "runtime/lib/python*/site-packages/*.dist-info/WHEEL"
        ):
            versions = re.findall(r"macosx_(\d+)_(\d+)", wheel.read_text())
            if versions:
                minimums.append(min((int(a), int(b)) for a, b in versions))
        minimum = ".".join(map(str, max(minimums)))
        run(
            "clang",
            "-fobjc-arc",
            "-Os",
            "-mmacosx-version-min=" + minimum,
            "-framework",
            "Cocoa",
            HERE / "launcher.m",
            "-o",
            binaries / "AutoGrowth",
        )
        iconset = temp / "AutoGrowth.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                name = f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
                run(
                    "sips",
                    "-z",
                    size * scale,
                    size * scale,
                    HERE / "logo.png",
                    "--out",
                    iconset / name,
                    stdout=subprocess.DEVNULL,
                )
        run("iconutil", "-c", "icns", iconset, "-o", resources / "AutoGrowth.icns")
        version = (Path(payload) / "AutoGrowth/VERSION").read_text().strip()
        with (contents / "Info.plist").open("wb") as file:
            plistlib.dump(
                {
                    "CFBundleName": "AutoGrowth",
                    "CFBundleDisplayName": "AutoGrowth",
                    "CFBundleIdentifier": "org.biosustain.autogrowth",
                    "CFBundleExecutable": "AutoGrowth",
                    "CFBundlePackageType": "APPL",
                    "CFBundleIconFile": "AutoGrowth.icns",
                    "CFBundleShortVersionString": version,
                    "CFBundleVersion": version,
                    "LSMinimumSystemVersion": minimum,
                    "NSHighResolutionCapable": True,
                    "NSPrincipalClass": "NSApplication",
                },
                file,
            )
        shutil.move(str(payload), str(resources / "payload"))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(app), str(output))
    return output


def make_dmg(app, destination):
    if Path(destination).exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix="autogrowth-dmg-") as temp:
        stage = Path(temp)
        shutil.copytree(app, stage / "AutoGrowth.app", symlinks=True)
        (stage / "Applications").symlink_to("/Applications")
        run(
            "hdiutil",
            "create",
            "-volname",
            "AutoGrowth",
            "-srcfolder",
            stage,
            "-format",
            "UDZO",
            "-o",
            destination,
        )


def wrap_windows(payload):
    payload = Path(payload)
    with tempfile.TemporaryDirectory(prefix="autogrowth-native-") as temp:
        temp = Path(temp)
        run(
            payload / "runtime/python.exe",
            "-s",
            "-B",
            "-c",
            "from PIL import Image; import sys; Image.open(sys.argv[1]).save(sys.argv[2], sizes=[(16,16),(32,32),(48,48),(256,256)])",
            HERE / "logo.png",
            temp / "AutoGrowth.ico",
        )
        (temp / "launcher.rc").write_text('1 ICON "AutoGrowth.ico"\n', encoding="utf-8")
        run(
            "rc",
            "/nologo",
            "/fo",
            temp / "launcher.res",
            temp / "launcher.rc",
            cwd=temp,
        )
        run(
            "cl",
            "/nologo",
            "/O2",
            "/MT",
            HERE / "launcher.c",
            temp / "launcher.res",
            "user32.lib",
            "/link",
            "/SUBSYSTEM:WINDOWS",
            "/OUT:" + str(payload / "AutoGrowth.exe"),
            cwd=temp,
        )
        shutil.copy2(temp / "AutoGrowth.ico", payload / "AutoGrowth.ico")


def make_installer(payload, output, bootstrapper, iscc="ISCC.exe"):
    # Require a separately acquired, signed Microsoft Evergreen bootstrapper.
    bootstrapper = Path(bootstrapper).resolve()
    if not bootstrapper.is_file():
        raise FileNotFoundError(bootstrapper)
    env = os.environ.copy()
    env["AUTOGROWTH_WEBVIEW_BOOTSTRAPPER"] = str(bootstrapper)
    run(
        "pwsh",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "$s=Get-AuthenticodeSignature -LiteralPath $env:AUTOGROWTH_WEBVIEW_BOOTSTRAPPER; "
        "if ($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { exit 1 }",
        env=env,
    )
    version = (Path(payload) / "AutoGrowth/VERSION").read_text().strip()
    run(
        iscc,
        "/DAppVersion=" + version,
        "/DBundleDir=" + str(Path(payload).resolve()),
        "/DOutputDir=" + str(Path(output).resolve()),
        "/DWebViewBootstrapper=" + str(bootstrapper),
        HERE / "windows.iss",
    )
