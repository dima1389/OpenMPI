@echo off
setlocal

rem Prepend MinGW-w64's bin to PATH so its runtime DLLs are found first
set "PATH=C:\msys64\mingw64\bin;%PATH%"

rem Define clean MSMPI paths WITHOUT trailing backslash
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

cd /d %~dp0

rem Build MPI program with MinGW gcc + MSMPI
echo Building MPI_Parallel_Sum_Block...
gcc MPI_Parallel_Sum_Block.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Parallel_Sum_Block.exe

call mpiexec -n 4 MPI_Parallel_Sum_Block.exe

endlocal
