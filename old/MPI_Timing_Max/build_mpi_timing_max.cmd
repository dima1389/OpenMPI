@echo off
setlocal

rem ============================================================
rem  MPI Build Script (MinGW + Microsoft MPI)
rem  Example: Measuring local and maximum execution time
rem ============================================================

rem --- Ensure MinGW DLLs come first (avoids crc32_combine popup) ---
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%SystemRoot%\system32;%PATH%"

rem --- Microsoft MPI include and library paths (NO trailing backslash) ---
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

rem --- Move to script directory ---
cd /d "%~dp0"

rem --- Source and output names ---
set "SRC=MPI_Timing_Max.c"
set "OUT=MPI_Timing_Max.exe"

echo.
echo ============================================================
echo Building %SRC%
echo ============================================================

g++ "%SRC%" ^
    -I"%MSMPI_INC%" ^
    -L"%MSMPI_LIB64%" ^
    -lmsmpi ^
    -Wall -Wextra ^
    -O2 ^
    -o "%OUT%"

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Build successful.
echo ============================================================

echo.
echo ============================================================
echo Runtime behavior:
echo   - Each MPI process measures its own execution time
echo   - All local times are reduced using MPI_MAX
echo   - Rank 0 prints the maximum (slowest process) duration
echo ============================================================

echo Running MPI program with 4 processes...
echo.

call mpiexec -n 4 "%OUT%"

echo.
echo ============================================================
echo Done.
echo ============================================================

endlocal
