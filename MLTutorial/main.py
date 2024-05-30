import sys
import os

# Add the parent directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import modules from the 'src' folder
from src.logger import logging

def main():
    logging.info("Logging from main.py")

if __name__ == "__main__":
    main()
