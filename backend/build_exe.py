"""
Script to build the server as a standalone EXE using PyInstaller
"""
import PyInstaller.__main__
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# PyInstaller arguments
args = [
    'server.py',
    '--name=BrightSpotDetectorServer',
    '--onefile',  # Create a single executable file
    '--console',  # Keep console window (change to --noconsole if you want no window)
    '--clean',    # Clean PyInstaller cache before building
    
    # Hidden imports for dependencies
    '--hidden-import=ultralytics',
    '--hidden-import=ultralytics.models',
    '--hidden-import=ultralytics.utils',
    '--hidden-import=ultralytics.trackers',
    '--hidden-import=lapx',  # Required by ultralytics trackers
    '--hidden-import=cv2',
    '--hidden-import=flask',
    '--hidden-import=flask_cors',
    '--hidden-import=pydantic',
    '--hidden-import=torch',
    '--hidden-import=torchvision',
    '--hidden-import=numpy',
    '--hidden-import=PIL',
    '--hidden-import=PIL.Image',
    
    # Collect all submodules (important for ultralytics and torch)
    '--collect-all=ultralytics',
    '--collect-all=torch',
    '--collect-all=torchvision',
    '--collect-all=cv2',
    
    # Include data files if needed (models directory)
    # Uncomment and adjust path if you want to bundle models:
    # f'--add-data={os.path.join(project_root, "Models")};Models',
    
    # Output paths
    '--distpath=dist',
    '--workpath=build',
    '--specpath=.',
    
    # Exclude unnecessary modules to reduce size (optional)
    # '--exclude-module=matplotlib',
    # '--exclude-module=IPython',
]

# Change to script directory
os.chdir(script_dir)

print("Starting PyInstaller build...")
print("This may take several minutes due to PyTorch and YOLO dependencies...")

# Run PyInstaller
try:
    PyInstaller.__main__.run(args)
    print("\n" + "="*60)
    print("Build complete! EXE should be in the 'dist' directory.")
    print("="*60)
    print("\nNote: The EXE will be large (500MB-2GB) due to PyTorch and YOLO dependencies.")
    print("The executable is standalone and can be run on any Windows machine.")
except Exception as e:
    print(f"\nError during build: {e}")
    sys.exit(1)