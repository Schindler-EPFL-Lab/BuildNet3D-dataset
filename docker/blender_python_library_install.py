import ensurepip
import subprocess
import sys

ensurepip.bootstrap()
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
)
