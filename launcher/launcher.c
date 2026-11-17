/*
 * WhisperBlocks.exe - tiny launcher for the WhisperBlocks GUI.
 *
 * It contains no Python and no application logic. It only:
 *   1. finds the project folder (the folder this exe is in);
 *   2. offers to run install.bat when the .venv environment is missing;
 *   3. offers to install the GUI packages (requirements-gui.txt) when missing;
 *   4. starts  .venv\Scripts\pythonw.exe src\gui_main.py [files...]  and exits.
 *
 * Build (MinGW-w64), see launcher/build.sh:
 *   x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher.res
 *   x86_64-w64-mingw32-gcc -O2 -s -municode -mwindows launcher.c launcher.res -o WhisperBlocks.exe
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wchar.h>

#define BIG 32768
#define TITLE L"WhisperBlocks"

static wchar_t g_root[BIG];
static wchar_t g_buffer[BIG];
static wchar_t g_command[BIG];
static int g_ukrainian = 0;

static const wchar_t *tr(const wchar_t *uk, const wchar_t *en) { return g_ukrainian ? uk : en; }

static BOOL exists(const wchar_t *relative)
{
    if (_snwprintf(g_buffer, BIG, L"%ls\\%ls", g_root, relative) < 0) return FALSE;
    g_buffer[BIG - 1] = 0;
    return GetFileAttributesW(g_buffer) != INVALID_FILE_ATTRIBUTES;
}

/* Runs "cmd.exe /c <script>" in its own console inside the project folder and waits. */
static BOOL run_console_and_wait(const wchar_t *script)
{
    STARTUPINFOW startup;
    PROCESS_INFORMATION process;
    ZeroMemory(&startup, sizeof startup);
    startup.cb = sizeof startup;
    if (_snwprintf(g_command, BIG, L"cmd.exe /c \"%ls\"", script) < 0) return FALSE;
    g_command[BIG - 1] = 0;
    if (!CreateProcessW(NULL, g_command, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, g_root, &startup, &process))
        return FALSE;
    WaitForSingleObject(process.hProcess, INFINITE);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return TRUE;
}

/* Everything after the program name, exactly as typed (keeps the user's quoting). */
static const wchar_t *argument_tail(void)
{
    const wchar_t *p = GetCommandLineW();
    if (*p == L'"') {
        p++;
        while (*p && *p != L'"') p++;
        if (*p == L'"') p++;
    } else {
        while (*p && *p != L' ' && *p != L'\t') p++;
    }
    while (*p == L' ' || *p == L'\t') p++;
    return p;
}

static BOOL gui_packages_present(void)
{
    return exists(L".venv\\Lib\\site-packages\\PySide6") && exists(L".venv\\Lib\\site-packages\\tomlkit");
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show)
{
    STARTUPINFOW startup;
    PROCESS_INFORMATION process;
    wchar_t *slash;
    (void)instance; (void)previous; (void)arguments; (void)show;

    g_ukrainian = PRIMARYLANGID(GetUserDefaultUILanguage()) == LANG_UKRAINIAN;

    if (!GetModuleFileNameW(NULL, g_root, BIG - 1)) return 1;
    slash = wcsrchr(g_root, L'\\');
    if (!slash) return 1;
    *slash = 0;
    SetCurrentDirectoryW(g_root);

    if (!exists(L"src\\gui_main.py")) {
        MessageBoxW(NULL, tr(
            L"Не знайдено src\\gui_main.py.\n\nWhisperBlocks.exe має лежати в папці проєкту, поруч із run.bat та папкою src.",
            L"src\\gui_main.py was not found.\n\nWhisperBlocks.exe must stay in the project folder, next to run.bat and the src folder."),
            TITLE, MB_ICONERROR);
        return 1;
    }

    if (!exists(L".venv\\Scripts\\pythonw.exe")) {
        if (MessageBoxW(NULL, tr(
                L"Середовище Python (.venv) ще не встановлено.\n\n"
                L"Запустити встановлення зараз? Потрібен Python 3.12, інтернет і близько 10 ГБ на диску. "
                L"Це робиться один раз і триває 10–30 хвилин.",
                L"The Python environment (.venv) is not installed yet.\n\n"
                L"Run the setup now? It needs Python 3.12, internet access and about 10 GB of disk space. "
                L"This is done once and takes 10-30 minutes."),
                TITLE, MB_ICONQUESTION | MB_YESNO) != IDYES)
            return 1;
        if (!exists(L"install.bat") || !run_console_and_wait(L"install.bat") || !exists(L".venv\\Scripts\\pythonw.exe")) {
            MessageBoxW(NULL, tr(
                L"Встановлення не завершилось. Запустіть install.bat вручну й перегляньте повідомлення у вікні.",
                L"Setup did not complete. Run install.bat manually and read the messages in its window."),
                TITLE, MB_ICONERROR);
            return 1;
        }
    }

    if (!gui_packages_present()) {
        if (MessageBoxW(NULL, tr(
                L"Потрібно один раз встановити компоненти вікна програми (PySide6, близько 250 МБ).\n\nВстановити зараз?",
                L"The window components (PySide6, about 250 MB) need to be installed once.\n\nInstall now?"),
                TITLE, MB_ICONQUESTION | MB_YESNO) != IDYES)
            return 1;
        run_console_and_wait(L".venv\\Scripts\\python.exe -m pip install -r requirements\\requirements-gui.txt || pause");
        if (!gui_packages_present()) {
            MessageBoxW(NULL, tr(
                L"Не вдалося встановити компоненти вікна. Перевірте інтернет і спробуйте ще раз або запустіть run-gui.bat.",
                L"The window components could not be installed. Check the internet connection and try again, or run run-gui.bat."),
                TITLE, MB_ICONERROR);
            return 1;
        }
    }

    if (_snwprintf(g_command, BIG, L"\"%ls\\.venv\\Scripts\\pythonw.exe\" \"%ls\\src\\gui_main.py\" %ls",
                   g_root, g_root, argument_tail()) < 0) {
        MessageBoxW(NULL, tr(L"Занадто довгий список файлів.", L"The file list is too long."), TITLE, MB_ICONERROR);
        return 1;
    }
    g_command[BIG - 1] = 0;
    ZeroMemory(&startup, sizeof startup);
    startup.cb = sizeof startup;
    if (!CreateProcessW(NULL, g_command, NULL, NULL, FALSE, 0, NULL, g_root, &startup, &process)) {
        MessageBoxW(NULL, tr(
            L"Не вдалося запустити .venv\\Scripts\\pythonw.exe.",
            L"Could not start .venv\\Scripts\\pythonw.exe."), TITLE, MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return 0;
}
