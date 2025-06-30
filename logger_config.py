import logging
import os

def setup_logger(log_folder="log", log_filename="pipeline.log", log_level=logging.INFO):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_filename)
    if os.path.exists(log_path):
        os.remove(log_path)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Remove old handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    return root_logger
