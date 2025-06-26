import time
import logging
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f"{func.__name__} ran in {end - start:.4f} seconds")
        return result
    return wrapper
