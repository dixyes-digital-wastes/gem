@echo off
call buildconf.bat
if errorlevel 1 exit /b 1
call configure.bat --disable-all --enable-cli --enable-zts --enable-toy=shared --enable-swow=shared --disable-swow-ssl --disable-swow-curl --disable-swow-pdo-pgsql --disable-swow-mongodb
if errorlevel 1 exit /b 1
nmake
if errorlevel 1 exit /b 1
exit /b 0
