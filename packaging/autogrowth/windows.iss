; Build using Inno Setup 6.3+ on Windows x64. Paths are supplied by build.py.
#ifndef BundleDir
  #error BundleDir is required
#endif
#ifndef WebViewBootstrapper
  #error WebViewBootstrapper is required
#endif
#ifndef AppVersion
  #error AppVersion is required
#endif

[Setup]
AppId={{9DE62289-8E22-4DC8-9E1A-59F1107C659C}
AppName=AutoGrowth
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\AutoGrowth
DefaultGroupName=AutoGrowth
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
OutputDir={#OutputDir}
OutputBaseFilename=AutoGrowth-Setup-x64
SetupIconFile={#BundleDir}\AutoGrowth.ico
UninstallDisplayIcon={app}\AutoGrowth.exe
Compression=lzma2
SolidCompression=yes
AppMutex=Local\AutoGrowth.Desktop
CloseApplications=no
RestartApplications=no
WizardStyle=modern

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#WebViewBootstrapper}"; DestDir: "{tmp}"; DestName: "MicrosoftEdgeWebview2Setup.exe"; Flags: deleteafterinstall; Check: NeedsWebView2

[InstallDelete]
; Prevent old package versions remaining after an upgrade. User data is elsewhere.
Type: filesandordirs; Name: "{app}\runtime"

[Icons]
Name: "{group}\AutoGrowth"; Filename: "{app}\AutoGrowth.exe"
Name: "{group}\Uninstall AutoGrowth"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\AutoGrowth.exe"; Description: "Launch AutoGrowth"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsWebView2: Boolean;
var
  Version: String;
  Key: String;
begin
  Key := 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  Result := not ((RegQueryStringValue(HKCU, Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
                 (RegQueryStringValue(HKLM32, Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
begin
  if (CurStep = ssPostInstall) and NeedsWebView2 then begin
    if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, Code) then
      RaiseException('Could not start the Microsoft WebView2 installer. Install WebView2 before launching AutoGrowth.');
    if (Code <> 0) or NeedsWebView2 then
      RaiseException('WebView2 installation failed. Internet access is required. Install Microsoft WebView2 before launching AutoGrowth.');
  end;
end;
