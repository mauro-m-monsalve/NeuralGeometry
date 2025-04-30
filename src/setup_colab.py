


import os
import subprocess
import sys
import re

def run(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

if not in_colab():
    print("ℹ️ Not in Google Colab — skipping Colab-specific setup.")
    sys.exit(0)

if not has_gpu():
    raise RuntimeError("❌ GPU is not available in this Colab runtime. Please enable GPU under Runtime > Change runtime type.")

# 1. Clone the repository if not already present
if not os.path.exists("NeuralGeometry"):
    run("git clone https://github.com/mauro-m-monsalve/NeuralGeometry.git")

# 2. Change working directory
os.chdir("NeuralGeometry")

# 3. Parse environment.yml if present and try pip-installing packages
env_path = "environment.yml"
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" in line:
                continue
            # Convert conda-style to pip-style specifier if needed
            package = re.sub(r"=+", "==", line)
            try:
                run(f"{sys.executable} -m pip install --quiet {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install: {package}")

print("✅ Colab GPU setup complete.")