import importlib.util
import subprocess
import sys

def ensure_spacy_model(model_name):
    if importlib.util.find_spec(model_name) is None:
        print(f"Model '{model_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    else:
        print(f"Model '{model_name}' is already installed.")
