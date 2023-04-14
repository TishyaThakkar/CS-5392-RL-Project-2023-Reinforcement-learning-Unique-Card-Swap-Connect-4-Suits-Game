@echo off

setlocal

:thefirst

echo Choose an option to train:
echo 1. Train Agent Iron
echo 2. Train Agent Gold
echo 3. Train Agent Diamond1
echo 4. Train Agent Diamond2

SET /P AGENT=Choose a model to train(1,2,3,4): 


if %AGENT% == 1 (
  cd .\Iteration1\
  call .\train.bat
  exit /B
)

if %AGENT% == 2 (
   cd .\Iteration2
  call .\train.bat
  exit /B
)

if %AGENT% == 3 (
  cd .\Iteration3_1\
  call .\train.bat
  exit /B
)

if %AGENT% == 4 (
  cd .\Iteration3_2\
  call .\train.bat
  exit /B
)

ECHO Unkown option 
goto thefirst