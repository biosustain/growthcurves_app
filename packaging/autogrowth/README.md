# Portable AutoGrowth

One standalone CPython runtime of AutoGrowth app. Uses the user's existing browser.
Runtime packages include only the Excel engines needed for XLS/XLSX uploads and XLSX
exports (not pandas' complete Excel extra). Transitive dependencies required by
growthcurves and Streamlit remain included.

## Build

Install uv on the build machine. Run on the target OS and architecture:

```sh
python packaging/autogrowth/build.py --output /absolute/path/to/AutoGrowth
```

The supplied macOS arm64 and Windows x64 hashed locks pin runtime dependencies. Use
`--refresh-lock` only to update them (or create a lock for macOS Intel), then review and
commit the changed lock. The output must not exist. Use `--uv /path/to/uv` if uv is not
on PATH. Windows example (from the repository root):

```powershell
py packaging/autogrowth/build.py --output C:\builds\AutoGrowth
```

Python 3.12.11 is pinned and downloaded using uv's python-build-standalone runtime,
rather than copying a virtual environment. Downloads and dependency installation require
internet during building. Native scientific dependencies require wheels; growthcurves may
build from its hashed source distribution in a temporary build environment. Build on
macOS arm64 for Apple Silicon, macOS x86_64 for Intel, or Windows x64. Cross-building is
intentionally rejected. Windows must be exercised on Windows. No signing, notarization,
installer creation, or publishing is performed. See
https://docs.astral.sh/uv/concepts/python-versions/#managed-python-distributions.

## Run

Move the whole folder anywhere, including a path with spaces. Double-click
`AutoGrowth.command` on macOS or `AutoGrowth.cmd` on Windows. No Python, uv, Cargo, or
dependency installation is needed on the user's machine. The launcher opens the system
browser after server readiness, binds only to 127.0.0.1 on a free port, and retains a
terminal window. Press Ctrl+C in that window to stop the server. Closing the browser tab
does not stop it. Logs are in the OS temporary directory under `AutoGrowth-logs`. Use
`--no-browser` for manual browser launch, or `--check` to check imports/assets.

Only the three example CSVs referenced by the upload page are included. External links in
the help still require internet. Local data analysis requires no model downloads. The
source contains its own logo and packaged help illustrations.

## Validation and size

The builder checks imports, method notes, page rendering, a representative fit, peak
detection, and Excel export/import after relocating to a path with spaces and restricting
PATH. `build-report.json` records the target and directory sizes. This is not a full
UI/analysis test: before release, exercise upload, batch fitting, turbidostat analysis,
download, and shutdown on a clean target machine. Windows results are not implied by
macOS results.

Bytecode caches, static `.a` libraries, and test suites in an explicit scientific-package
allowlist are removed. Runtime native libraries, package data, metadata, and licenses
remain intact. Do not indiscriminately remove package subdirectories to save space:
SciPy, matplotlib and Streamlit load resources dynamically. Archive the directory for
transfer; an archive does not replace a signed application or installer.

Run the included smoke checks with the bundled interpreter:

```sh
/path/to/AutoGrowth/runtime/bin/python3 -s -B packaging/autogrowth/verify.py /path/to/AutoGrowth
```

On Windows use `AutoGrowth\runtime\python.exe`. These checks render pages with Streamlit
AppTest, run a representative fit, and round-trip an Excel export.

## Validation in this checkout

macOS arm64: relocated imports, all seven file-based pages, representative sliding-window
fitting, Excel export/import, HTTP readiness, and launcher/server shutdown passed.
Windows x64: dependency lock resolved; artifact has not been built or exercised on a
Windows host. Clean-machine testing remains outstanding.
