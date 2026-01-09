@echo off
setlocal

rem --------------------------------------------------------------------
rem  Modify PATH so MinGW DLLs are used first (prevents popup issues)
rem --------------------------------------------------------------------
set "PATH=C:\msys64\mingw64\bin;%PATH%"

rem --------------------------------------------------------------------
rem  Define Microsoft MPI include & library folders (NO trailing '\')
rem --------------------------------------------------------------------
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

rem --------------------------------------------------------------------
rem  Move to directory where the script is located
rem --------------------------------------------------------------------
cd /d %~dp0

rem --------------------------------------------------------------------
rem  Check arguments
rem  Usage: build_mpi_matvec.cmd vector.txt matrix.txt [num_procs]
rem --------------------------------------------------------------------
@REM if "%~1"=="" (
@REM     echo [ERROR] Missing vector file.
@REM     echo Usage: %~nx0 vector.txt matrix.txt [num_procs]
@REM     exit /b 1
@REM )

@REM if "%~2"=="" (
@REM     echo [ERROR] Missing matrix file.
@REM     echo Usage: %~nx0 vector.txt matrix.txt [num_procs]
@REM     exit /b 1
@REM )

@REM set "VEC_FILE=%~1"
@REM set "MAT_FILE=%~2"

set "VEC_FILE=Vector.txt"
set "MAT_FILE=Matrix.txt"

rem Optional: number of MPI processes (default = 4)
if "%~3"=="" (
    set NP=4
) else (
    set NP=%~3
)

rem --------------------------------------------------------------------
rem  Build the MPI matrix-vector program
rem --------------------------------------------------------------------
echo Building MPI_Matrix_Vector_General.c ...
gcc MPI_Matrix_Vector_General.c ^
  -I"%MSMPI_INC%" ^
  -L"%MSMPI_LIB64%" ^
  -lmsmpi ^
  -o MPI_Matrix_Vector_General.exe

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo Build completed successfully.

rem --------------------------------------------------------------------
rem  Run the MPI program
rem --------------------------------------------------------------------
echo Running: mpiexec -n %NP% MPI_Matrix_Vector_General.exe %VEC_FILE% %MAT_FILE%
echo --------------------------------------------------------------
call mpiexec -n %NP% MPI_Matrix_Vector_General.exe "%VEC_FILE%" "%MAT_FILE%"

endlocal
