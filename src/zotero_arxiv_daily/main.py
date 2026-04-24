import os
import sys
import logging
from omegaconf import DictConfig
import hydra
from loguru import logger
import dotenv
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.classics import _as_bool
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config:DictConfig):
    debug_enabled = _as_bool(config.executor.debug)
    # Configure loguru log level based on config
    log_level = "DEBUG" if debug_enabled else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    for logger_name in logging.root.manager.loggerDict:
        if "zotero_arxiv_daily" in logger_name:
            continue
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if debug_enabled:
        logger.info("Debug mode is enabled")
    
    executor = Executor(config)
    executor.run()

if __name__ == '__main__':
    main()
