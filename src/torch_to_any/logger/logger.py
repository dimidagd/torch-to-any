import logging

import coloredlogs

# Setup logger
logger = logging.getLogger("onnx_export")
logger.setLevel(logging.INFO)

# Setup coloredlogs
coloredlogs.install(level=logging.INFO, logger=logger, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
