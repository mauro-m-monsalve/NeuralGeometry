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
    print("❌ GPU is not available in this Colab runtime. Please enable GPU under Runtime > Change runtime type.")
    sys.exit(1)

# 1. Clone the repository if not already present
if not os.path.exists("NeuralGeometry"):
    run("git clone https://github.com/mauro-m-monsalve/NeuralGeometry.git")

# 2. Change working directory
os.chdir("NeuralGeometry")

# 3. Parse environment.yml if present and try pip-installing packages
env_path = "environment.yml"
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        deps_section = False
        for line in f:
            line = line.strip()
            if line.startswith("dependencies:"):
                deps_section = True
                continue
            if deps_section:
                if line.startswith("- "):
                    pkg = line[2:].strip()
                    if not pkg or pkg.startswith("python=") or pkg.startswith("#"):
                        continue
                    pip_pkg = re.sub(r"=+", "==", pkg)
                    try:
                        run(f"{sys.executable} -m pip install --quiet {pip_pkg}")
                    except subprocess.CalledProcessError:
                        print(f"⚠️ Failed to install: {pip_pkg}")
                elif not line.startswith(" "):
                    break  # Stop if we reach another section

print("✅ Colab GPU setup complete.")