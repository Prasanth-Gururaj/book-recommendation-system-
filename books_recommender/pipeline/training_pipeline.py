import logging
import sys
from books_recommender.components.stage_00_data_ingestion import DataIngestion

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()

    def start_training_pipeline(self):
        try:
            # Start the data ingestion process
            self.data_ingestion.initiate_data_ingestion()
            logging.info("Training pipeline started successfully.")
        except Exception as e:
            raise Exception(e, sys) from e