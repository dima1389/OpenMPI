@echo off
setlocal

rem ============================================================
rem  MPI Build Script (MinGW + Microsoft MPI)
rem  Example: Broadcasting a struct using MPI_Type_create_struct
rem ============================================================

rem Ensure MinGW DLLs are found first (avoids crc32_combine popup)
set "PATH=C:\msys64\mingw64\bin;%PATH%"

rem Microsoft MPI include and library paths (NO trailing backslash)
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

rem Move to script directory
cd /d "%~dp0"

rem Source and output names
set "SRC=MPI_Bcast_Struct.c"
set "OUT=MPI_Bcast_Struct.exe"

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
echo Runtime input required (ENTER ONCE, ONLY ON RANK 0):
echo   ^<int^> ^<double^> ^<double^>
echo Example:
echo   10 3.14 2.718
echo ============================================================

echo Running MPI program with 4 processes...
echo.

call mpiexec -n 4 "%OUT%"

echo.
echo ============================================================
echo Done.
echo ============================================================

endlocal
