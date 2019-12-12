## Setup vscode for windows

* Install python 64bit 
    1. [Download Page](https://www.python.org/downloads/windows/)
    2. Select "Download Windows x86-64 executable installer" . Do not install the 32 bit version
    3. After running the downloaded installed remeber to enalbe " ADD PYTHON 3.8 TO PATH"
* Add
    * python extension vscode
    * edit csv extension
    * markdown preview enhanced
    * extra: material icon theme
* Set python interpreter. Press CTRL-SHIFT-P and type Python Select Interpreter
    * My interpreter location is : C:\Users\manue\AppData\Local\Programs\Python\Python38
* Open a terminal and install packages using pip
    ```
    pip install scikit-learn==0.21.3 # install scikit-learn (do not install latest verison)
    pip install pandas # install pandas dataframe
    pip install pylint # automatic linking vscode python
    pip install matplotlib
    ```
    Note: There is an bug with version 0.22 scikit-learn 
    

## Useful reference
[Install Python Windows](https://code.visualstudio.com/docs/languages/python)   