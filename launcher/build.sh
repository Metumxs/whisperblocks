#!/bin/sh
# Rebuilds WhisperBlocks.exe from launcher.c with MinGW-w64 (Linux, WSL or MSYS2).
#   Debian/Ubuntu/WSL:  sudo apt install gcc-mingw-w64-x86-64
#   MSYS2 (UCRT64):     pacman -S mingw-w64-ucrt-x86_64-gcc   (then use plain gcc/windres)
set -eu
cd "$(dirname "$0")"
CC=${CC:-x86_64-w64-mingw32-gcc}
WINDRES=${WINDRES:-x86_64-w64-mingw32-windres}
"$WINDRES" launcher.rc -O coff -o launcher.res
"$CC" -O2 -s -Wall -Wextra -municode -mwindows launcher.c launcher.res -o ../WhisperBlocks.exe
rm -f launcher.res
ls -l ../WhisperBlocks.exe
