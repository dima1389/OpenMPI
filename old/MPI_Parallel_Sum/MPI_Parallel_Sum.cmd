@echo off
setlocal

rem Make sure MinGW’s DLLs come first so there is no crc32_combine popup
set "PATH=C:\msys64\mingw64\bin;%PATH%"

rem Define clean MSMPI paths WITHOUT trailing backslash
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

cd /d %~dp0

rem Build MPI program with MinGW g++ + MSMPI
echo Building MPI_Parallel_Sum...
gcc MPI_Parallel_Sum.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Parallel_Sum.exe

call mpiexec -n 4 MPI_Parallel_Sum.exe

endlocal
