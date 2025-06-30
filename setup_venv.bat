@echo off
echo Setting up clean virtual environment...

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

:: Create necessary directories
mkdir data 2>nul
mkdir models 2>nul
mkdir logs 2>nul

:: Copy .env.example to .env if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env file from .env.example
    echo Please update .env with your actual API keys and configuration
)

echo Virtual environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat 