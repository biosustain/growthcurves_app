#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR commandLine, int show) {
    wchar_t root[32768], python[32768], command[32768];
    DWORD n = GetModuleFileNameW(NULL, root, 32768);
    if (!n || n >= 32768) return 1;
    wchar_t *slash = wcsrchr(root, L'\\');
    if (!slash) return 1;
    *slash = 0;
    if (swprintf_s(python, 32768, L"%ls\\runtime\\pythonw.exe", root) < 0 ||
        swprintf_s(command, 32768, L"\"%ls\" -s -B \"%ls\\bootstrap.py\"", python, root) < 0) return 1;
    SetEnvironmentVariableW(L"PYTHONHOME", NULL);
    SetEnvironmentVariableW(L"PYTHONPATH", NULL);
    STARTUPINFOW startup = {0}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION child = {0};
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, root, &startup, &child)) {
        MessageBoxW(NULL, L"The bundled Python runtime could not be launched. Reinstall the complete application.", L"AutoGrowth", MB_ICONERROR);
        return 1;
    }
    CloseHandle(child.hThread);
    // The Python GUI owns shutdown and the single-instance mutex. This stub exits.
    CloseHandle(child.hProcess);
    return 0;
}
