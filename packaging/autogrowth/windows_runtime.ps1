#Requires -Version 7.0
<# Build-time helper: obtain and verify Python.org's embeddable runtime.
   No elevation, execution-policy changes, or end-user PowerShell launcher.
#>
param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][ValidateSet("3.12.10")][string]$Version,
    [switch]$VerifyOnly
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $IsWindows) { throw "Build the Windows runtime on Windows." }

if (-not $VerifyOnly) {
    if (Test-Path -LiteralPath $Destination) { throw "Runtime destination already exists: $Destination" }
    $archive = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString() + '.zip')
    try {
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip" -OutFile $archive
        $expected = '4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3'
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
            throw "Python embeddable archive checksum mismatch."
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $Destination
    } finally {
        if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    }
}

# Check the actual launchers and interpreter DLLs before executing Python.
# This validates Python's publisher, not AutoGrowth's source/launcher/installer.
foreach ($name in @('python.exe', 'pythonw.exe', 'python3.dll', 'python312.dll')) {
    $path = Join-Path $Destination $name
    $signature = Get-AuthenticodeSignature -LiteralPath $path
    if ($signature.Status -ne 'Valid' -or
        $null -eq $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O="?Python Software Foundation"?(,|$)') {
        throw "Expected a valid Python Software Foundation signature on $name; got $($signature.Status)."
    }
    Write-Host "Verified Python Software Foundation signature: $name"
}

if (-not $VerifyOnly) {
    New-Item -ItemType Directory -Path (Join-Path $Destination 'Lib\site-packages') -Force | Out-Null
    # Paths are relative to runtime/, so the complete folder remains relocatable.
    # '..' exposes bootstrap's sibling modules; isolated Python ignores PYTHONPATH.
    @('python312.zip', '.', 'Lib\site-packages', '..', '..\src', '..\AutoGrowth', 'import site') |
        Set-Content -LiteralPath (Join-Path $Destination 'python312._pth') -Encoding ascii
}
