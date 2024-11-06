import os

from dotenv import load_dotenv
load_dotenv()

class Config:
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    INPUT_DIR = os.getenv("INPUT_DIR")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")

if __name__ == "__main__":
    print(f"PROJECT_DIR: {Config.PROJECT_DIR}")
    print(f"INPUT_DIR: {Config.INPUT_DIR}")
    print(f"OUTPUT_DIR: {Config.OUTPUT_DIR}")
