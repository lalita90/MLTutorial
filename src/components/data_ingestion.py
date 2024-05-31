import os
import sys
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Set up logging to save messages to a file
logging.basicConfig(filename='/Users/lmeena/Desktop/ML_Lalita/MLTutorial/logs/example.log', level=logging.INFO)

# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of 'src' to the Python path
src_parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(src_parent_dir)


# Import logger from the 'src' folder
from logger import logging as custom_logging
from exception import CustomException
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        custom_logging.info("Entered the data ingestion method or component")
        #try:
        df = pd.read_csv('notebook/data/stud.csv')
        custom_logging.info('Read the dataset as dataframe')

        # Check if the directory exists, if not create it
        artifacts_dir = os.path.dirname(self.ingestion_config.raw_data_path)
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

        custom_logging.info("Train test split initiated")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        custom_logging.info("Ingestion of the data is completed")

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
        #except CustomException as e:
         #   custom_logging.error("Error while ingesting data: %s", str(e))
          #  raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
