import logging

# Setup logger
logger = logging.getLogger("onnx_export")
logger.setLevel(logging.INFO)

# Add a StreamHandler if it doesn't exist
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
