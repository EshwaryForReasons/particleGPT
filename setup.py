
import subprocess
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure build directory exists
os.makedirs(os.path.join(script_dir, 'pTokenizer', 'build'), exist_ok=True)

# Build python module
os.chdir(os.path.join(script_dir, 'pTokenizer'))
subprocess.run("python setup.py bdist_wheel", shell=True)
subprocess.run("pip install dist/pTokenizer-0.0.1-cp311-cp311-linux_x86_64.whl --force-reinstall", shell=True)
