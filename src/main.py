import os
from pathlib import Path

from scripts import kaggle_02
from utils import logger
from utils import context

directory = context.get_context(os.path.abspath(__file__))
logger_name = Path(__file__).stem
logging = logger.setup_logger(logger_name, f"{directory}/src/logs/main.log")

def main():
  kaggle_02.go()

if __name__ == "__main__":
  main()
