import os
import logging

logger = logging.getLogger(__name__)

WORK_DIR = os.getenv("WORK")
DATA_FOLDER = os.path.join(WORK_DIR, "data")
CACHED_DATA_FOLDER = os.path.join(WORK_DIR, "cached_data")


VOCABULARY_SIZE = 32_768
CONTEXT_SIZE = 512

# Salvamos o path do Cache par ao HuggingFace
os.environ["HF_HOME"] = CACHED_DATA_FOLDER
os.chdir(WORK_DIR)

logger.info("Working directory set: %s", os.getcwd())
