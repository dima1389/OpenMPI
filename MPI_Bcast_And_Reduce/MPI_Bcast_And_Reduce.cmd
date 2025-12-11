@echo off
setlocal

rem --- Make sure MinGW’s DLLs come first so there is no crc32_combine popup ---
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%SystemRoot%\system32"

rem --- Define clean MSMPI paths WITHOUT trailing backslash ---
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

cd /d %~dp0

rem --- Build MPI program with MinGW g++ + MSMPI ---
g++ MPI_Bcast_And_Reduce.c ^
  -I"%MSMPI_INC%" ^
  -L"%MSMPI_LIB64%" ^
  -lmsmpi ^
  -o MPI_Bcast_And_Reduce.exe

call mpiexec -n 4 MPI_Bcast_And_Reduce.exe

endlocal
