"""
Script to build the server as a standalone EXE using PyInstaller
"""
import PyInstaller.__main__
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# PyInstaller arguments
args = [
    'server.py',
    '--name=BrightSpotDetectorServer',
    '--onefile',
    '--windowed',  # No console window (use --noconsole if you want console)
    '--add-data=../Models;Models',  # Include models directory if needed
    '--hidden-import=ultralytics',
    '--hidden-import=cv2',
    '--hidden-import=flask',
    '--hidden-import=flask_cors',
    '--hidden-import=pydantic',
    '--collect-all=ultralytics',
    '--collect-all=torch',
    '--distpath=dist',
    '--workpath=build',
    '--specpath=.',
]

# Change to script directory
os.chdir(script_dir)

# Run PyInstaller
PyInstaller.__main__.run(args)

print("\nBuild complete! EXE should be in the 'dist' directory.")
print("Note: The EXE will be large due to PyTorch and YOLO dependencies.")

