@echo off

py --version
echo Wil je de dependencies installeren? (Y/N)
choice /c YN /n

if errorlevel 2 goto NO
if errorlevel 1 goto YES

:YES
echo Dependencies aan het installeren.
py -m pip install codecarbon
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
py -m pip install nltk
echo Installatie voltooid.
pause
goto END

:NO
echo Installeren van dependencies overgeslagen.
pause
goto END

:END
echo Programma's aan het uitvoeren
py naivebayesclassifier.py
py AI_main_training.py
py AI_main.py
echo Programma's uitgevoerd. dt zijn de delta tijden, emissions_x.csv besanden het energieverbruik.
pause