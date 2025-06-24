# config.py
import os
from dotenv import load_dotenv
from dynaconf import Dynaconf

try:

    GLOBAL_CONFIG = Dynaconf(
        settings_files=["config.yaml", ".secrets.yaml"],
        environments=True,
        env="development"  # or "production"
    )

except FileNotFoundError:
    print("ERROR: config.yaml not found")