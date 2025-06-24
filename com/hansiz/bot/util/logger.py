# logger.py
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


# Log to file
file_handler = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
file_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

