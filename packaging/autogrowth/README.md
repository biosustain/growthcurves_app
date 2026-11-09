# AutoGrowth desktop packaging

AutoGrowth ships with one standalone Python runtime, the shared helpers from `src`, and
the assets used by its pages. MicroGrowth, training runners, development packages, and
private experimental data are excluded. The desktop window uses pywebview with macOS
WebKit or Windows WebView2; it does not bundle Chromium, Electron, or Qt.

## macOS build

Build on the matching Mac architecture with Python, uv, and the Xcode command-line tools
installed. From the repository root:

```sh
python packaging/autogrowth/build.py --desktop --dmg \
  --output dist/AutoGrowth-desktop/AutoGrowth.app
```

Use `--uv /absolute/path/to/uv` when needed. Output paths must not already exist. The
result is `AutoGrowth.app` plus `AutoGrowth.dmg`, containing the app and an Applications
shortcut. Drag the app into Applications, then open it. Apple Silicon is built here;
Intel requires a matching host and its own dependency lock. The minimum macOS version is
derived from installed wheel tags (14.0 for this arm64 build), and the native launcher is
compiled for that same minimum.

This development artifact is not Developer ID signed or notarized. Distribution to other
Macs needs a signing/notarization pass covering the nested Python runtime, native
libraries, and application bundle, plus a clean-machine test. No signing credentials or
publishing are configured by these scripts.

## Windows x64 build

Use a Windows x64 host, Python, uv, Visual Studio Build Tools (x64 Native Tools Command
Prompt with `cl` and `rc` on PATH), and Inno Setup 6.3 or newer. Download the Evergreen
WebView2 bootstrapper from Microsoft's official site:
<https://developer.microsoft.com/microsoft-edge/webview2/>. The build checks its
Authenticode signature and Microsoft publisher.

```powershell
py packaging/autogrowth/build.py --desktop --installer `
  --output C:\builds\AutoGrowth `
  --webview2-bootstrapper C:\downloads\MicrosoftEdgeWebview2Setup.exe `
  --iscc "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

The output folder contains `AutoGrowth.exe`; its parent contains
`AutoGrowth-Setup-x64.exe`. The installer installs per user, creates Start-menu shortcuts
and an uninstaller, and refuses to update/uninstall a running instance. The supervisor
retains the installer mutex until server cleanup finishes, even when the GUI dies. No
administrator access is requested.

WebView2 is checked in the per-user and machine registries. If missing, the installer
runs Microsoft's signed bootstrapper. This step requires internet. An offline WebView2
installer variant is not implemented. Windows compilation, installer execution, WebView2
detection, and Job Object behavior must still be validated on Windows; macOS results do
not validate those code paths.

## Running and stopping

The native launcher resolves resources from its own location, never the caller's working
directory. Closing the desktop window or choosing Quit requests confirmation: running
analysis stops and unexported results are lost. The confirmation is always shown; this
version does not detect dirty session state or save it automatically. Downloads are
enabled in the embedded webview and external links open in the system browser.
Uploads/downloads still need an end-to-end native-window check before release.

The window process owns a supervisor through an anonymous pipe. Closing the pipe,
including process death, requests graceful shutdown. The supervisor sends EOF to a server
adapter that schedules `Server.stop()` on its asyncio loop. A 12-second supervisor
deadline kills a stuck server; a separate 15-second server watchdog handles loss of the
supervisor. Unix process groups and Windows kill-on-close Job Objects contain
descendants. Only app-owned processes are targeted. The server waits for assignment to
its Windows Job Object before importing application code.

This uses multiple processes from **one Python installation**, not extra environments or
training runners. The adapter uses Streamlit's internal Server API, so Streamlit is
pinned to 1.66.1. Updating it requires rerunning lifecycle and application tests. There
is no remotely accessible HTTP shutdown endpoint. The server binds to localhost on a free
port, and readiness requires a status file from its own process plus an HTTP health
check. Startup failure is shown in the window and written to a log.

Logs and caches are outside the installed app:

- macOS: `~/Library/Application Support/AutoGrowth/`
- Windows: `%LOCALAPPDATA%\AutoGrowth\`

Only one GUI instance is allowed. A second launch reports that AutoGrowth is already
running; it does not yet focus the existing window. Browser mode is explicit:

```sh
/path/to/payload/runtime/bin/python3 -s -B /path/to/payload/bootstrap.py --browser
```

The macOS payload is `AutoGrowth.app/Contents/Resources/payload`. On Windows use
`runtime\python.exe`. `--no-browser` starts headlessly and prints the URL. In these
modes, Ctrl+C stops the server; closing a browser tab does not. `--check` checks imports
and packaged help. Omitting `--desktop` from the builder produces a portable folder with
the same desktop bootstrap and command launchers.

## Dependencies and size

The per-platform hashed locks pin runtime packages. Use `--refresh-lock` deliberately
when updating dependencies or generating an Intel Mac lock. Python 3.12.11 is a
standalone runtime downloaded by uv, not a copied virtual environment. Native scientific
packages require wheels; growthcurves may build from its hashed source release. Build
tooling and caches are not shipped.

Only XLS/XLSX engines needed by the application are direct Excel dependencies.
Scientific-package test suites from an explicit allowlist, bytecode caches, and static
libraries are trimmed. Runtime resources, shared libraries, metadata, and licenses
remain. Three UI example CSVs and local help illustrations are included. The new arm64
payload was approximately 624 MiB before the wrapper/icon overhead; the scientific
libraries account for most of it.

## Checks

The builder checks imports and application smoke tests after relocation to a path with
spaces and with developer tools removed from PATH. Native builds also run the subprocess
lifecycle tests. Run them separately with the bundled Python:

```sh
/path/to/payload/runtime/bin/python3 -s -B packaging/autogrowth/verify.py /path/to/payload
/path/to/payload/runtime/bin/python3 -s -B packaging/autogrowth/test_lifecycle.py /path/to/payload
```

macOS arm64 checks completed: all seven file-based pages via AppTest, representative
sliding-window fitting, peak detection, Excel round-trip, native launcher import mode,
and six lifecycle scenarios (normal close, close during startup, owner death, supervisor
death, stalled server, and startup failure). Tests check process exit and port closure;
graceful cases also check the server's shutdown marker.

The native upload page was visually inspected; close confirmation, Cancel, confirmed
close, Cmd+Q, native SIGTERM, and forced GUI termination were exercised. A relocated
native launcher also passed readiness and shutdown with a restricted PATH. File
dialogs/downloads, active-analysis quit, clean-machine behavior, and all Windows
execution remain release checks. A native-session log exposed `KeyError: 0` in the
growthcurves ECOD detector with a datetime index; this path remains a release blocker.
The existing smoke tests check execution and a representative result, not comprehensive
numerical accuracy or visual correctness.


## What is what?

Files in this folder and what they do.

Needef for the build:

| File | Purpose |
|---|---|
| `build.py` | Builder entry point. Assembles the payload (runtime, dependencies, app, assets), runs the checks, and optionally wraps it as `.app`/`.dmg` or Windows installer. |
| `bootstrap.py` | Startup script copied into the payload. Starts the supervisor and the desktop window, or browser/headless mode. |
| `desktop_support.py` | Desktop-window helpers copied into the payload (pywebview window, close confirmation, downloads, external links). |
| `lifecycle.py` | Supervisor logic copied into the payload: pipe ownership, graceful shutdown, kill deadlines, Job Objects/process groups. |
| `server_runner.py` | Streamlit server adapter copied into the payload. Runs `Server.stop()` on EOF and writes the readiness status file. |
| `verify.py` | Import and application smoke tests, run by the builder on the relocated payload. |
| `test_lifecycle.py` | Subprocess lifecycle tests (close, owner/supervisor death, stalled server, startup failure), run by native builds. |
| `native_package.py` | Wraps the payload into `AutoGrowth.app` and `AutoGrowth.dmg` (macOS) or `AutoGrowth.exe` and the installer (Windows). |
| `launcher.m` | macOS native launcher (Objective-C), compiled with `clang`. |
| `launcher.c` | Windows native launcher (C), compiled with `cl`. |
| `windows.iss` | Inno Setup script for `AutoGrowth-Setup-x64.exe`. |
| `logo.png` | Source image for the macOS icon set and the Windows `.ico`. |
| `requirements.in` | Direct runtime dependencies. |

Not needed for the build:

| File | Purpose |
|---|---|
| `README.md` | This documentation. |
| `requirements-macos-arm64.lock` | Hashed dependency lock for macOS arm64. |
| `requirements-windows-x64.lock` | Hashed dependency lock for Windows x64. |

The builder also reads files outside this folder, which must be committed too:
`AutoGrowth/app.py`, `AutoGrowth/logo.svg`, `AutoGrowth/markdowns/`,
`AutoGrowth/.streamlit/config.toml`, the example CSVs under `AutoGrowth/data/` listed in
`EXAMPLES` in `build.py`, and the shared helpers in `src/`.

## GitHub release automation

`.github/workflows/autogrowth-release.yml` runs on every push to `main` and through
Actions → AutoGrowth desktop release → Run workflow. Manual runs build the selected ref.
macOS Apple Silicon (`macos-15`) and Windows x64 (`windows-2022`) from the committed
dependency lock and run the builder's relocated smoke and lifecycle tests. Both must
succeed before publication; interactive installer/UI checks remain separate. Windows
installs Inno Setup if needed and downloads Microsoft's bootstrapper, whose signature is
checked by the builder.

Each successful run publishes a prerelease tagged
`autogrowth-build-<run-id>-<attempt>`, targeting the exact built commit. It attaches
`AutoGrowth-macos-arm64.zip`, `AutoGrowth-macos-arm64.dmg`,
`AutoGrowth-windows-x64.zip`, `AutoGrowth-Setup-x64.exe`, and `SHA256SUMS.txt`.
The macOS ZIP preserves app permissions and symlinks; the Windows ZIP contains the
portable app folder. Uploads are staged in a draft before publication. Failed uploads
leave a draft for inspection, and reruns use a new attempt tag. Existing stable
releases and the Latest designation are unchanged. Assets are also retained as
workflow artifacts for 14 days.

The release job uses the built-in `GITHUB_TOKEN` with `contents: write`; no personal
token or signing secrets are required. Repository policy must permit that permission.
The existing Python/PyPI workflow remains independent.


## Terminal fallback downloads

Every workflow build also saves `AutoGrowth-terminal-macos-arm64.zip` and
`AutoGrowth-terminal-windows-x64.zip`. These contain standalone folders with the
bundled Python runtime, independent of the native `.app`/DMG or installer.
Extract the complete folder and run `AutoGrowth.command` on macOS or
`AutoGrowth.cmd` on Windows. These terminal launchers explicitly use browser mode;
the native app launcher continues to use its embedded window. Keep the terminal
open while using AutoGrowth, and press Ctrl+C there to stop the server. Closing
the browser tab does not stop it. No system Python installation is needed.

Each fallback is built and smoke-tested separately, then uploaded as the
`autogrowth-terminal-<platform>` Actions artifact **before native packaging**.
It remains downloadable for 14 days even if a later native build fails. Successful
release runs additionally attach both terminal ZIPs and include their checksums.
The original desktop ZIPs, DMG, and Windows installer are still included.

This fallback does not fix signing/notarization or bypass macOS security checks.
The reported damaged-app warning has not yet been diagnosed or resolved.
