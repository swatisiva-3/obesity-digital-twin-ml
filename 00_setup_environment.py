"""
00_setup_environment.py
========================
Run this FIRST, once, before anything else:  python 00_setup_environment.py

What it does:
  1. Creates a Python virtual environment in ./venv (so this project's
     packages never collide with anything else on your machine).
  2. Installs every package this pipeline needs, pinned to versions known
     to work together, from requirements.txt.
  3. Prints the exact "activate" command for your OS so you can start
     running the numbered scripts (01_... through 07_...).

You only need to re-run this when requirements.txt changes -- not every
time you run the pipeline.

If you'd rather manage your own environment (conda, an existing venv,
etc.), you can skip this script and just run:
    pip install -r requirements.txt
"""

import os
import subprocess
import sys
import venv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, "venv")
REQUIREMENTS_FILE = os.path.join(PROJECT_DIR, "requirements.txt")


def create_virtual_environment():
    """Create ./venv if it doesn't already exist."""
    if os.path.exists(VENV_DIR):
        print(f"[setup] Virtual environment already exists at {VENV_DIR} -- skipping creation.")
        return
    print(f"[setup] Creating virtual environment at {VENV_DIR} ...")
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    print("[setup] Virtual environment created.")


def venv_python_path():
    """Return the path to the python executable inside ./venv, cross-platform."""
    if os.name == "nt":  # Windows
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def install_requirements():
    """pip install everything in requirements.txt, inside the venv."""
    py = venv_python_path()
    print(f"[setup] Upgrading pip inside the virtual environment ...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    print(f"[setup] Installing packages from {REQUIREMENTS_FILE} ...")
    subprocess.check_call([py, "-m", "pip", "install", "-r", REQUIREMENTS_FILE, "-q"])
    print("[setup] All packages installed successfully.")


def print_activation_instructions():
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    if os.name == "nt":
        print(r"Activate the environment with:   venv\Scripts\activate")
    else:
        print("Activate the environment with:   source venv/bin/activate")
    print("\nThen run the pipeline steps in order, e.g.:")
    print("   python 01_data_loader.py")
    print("   python 02_preprocessing.py")
    print("   python 03_scoring.py")
    print("   python 04_optuna_weight_learning.py")
    print("   python 05_train_model.py")
    print("   python 06_shap_explain.py --caseid 1703142")
    print("   python 07_phenograph_visualization.py")
    print("\n...or run everything at once with:   python run_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    create_virtual_environment()
    install_requirements()
    print_activation_instructions()
