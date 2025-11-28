from pathlib import Path

def get_path(number):
    return Path(__file__).resolve().parents[number]