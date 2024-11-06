import os

from dotenv import load_dotenv
load_dotenv("../.env")

class Config:
    DEBUG = eval(os.getenv("DEBUG", "False"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    PROJECT_DIR = os.getenv("PROJECT_DIR")
    assert os.path.isdir(PROJECT_DIR), f"PROJECT_DIR {PROJECT_DIR} is not a directory"

    INPUT_PATH = os.getenv("INPUT_PATH")
    assert os.path.exists(INPUT_PATH), f"INPUT_PATH {INPUT_PATH} does not exist"
    assert os.path.isfile(INPUT_PATH), f"INPUT_PATH {INPUT_PATH} is not a file"

    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    assert os.path.isdir(OUTPUT_DIR), f"OUTPUT_DIR {OUTPUT_DIR} is not a directory"

if __name__ == "__main__":
    print(f"DEBUG: {Config.DEBUG}")
    print(f"LOG_LEVEL: {Config.LOG_LEVEL}")
    print(f"PROJECT_DIR: {Config.PROJECT_DIR}")
    print(f"INPUT_PATH: {Config.INPUT_PATH}")
    print(f"OUTPUT_DIR: {Config.OUTPUT_DIR}")
