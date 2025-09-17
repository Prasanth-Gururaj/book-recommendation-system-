import os
import sys
import pickle
import time

from scipy.sparse import csr_matrix

from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException

class ModelEvaluation:

    def __init__(self, app_config = AppConfiguration()):
        try:
            self.model_evaluation_config = app_config.get_model_trainer_config()  
        except Exception as e:
            raise AppException(e, sys) from e

    def evaluate(self):
        try:
            # Load pivot data
            book_pivot = pickle.load(open(self.model_evaluation_config.transformed_data_file_dir, 'rb'))
            book_sparse = csr_matrix(book_pivot)

            # Load trained model
            file_name = os.path.join(self.model_evaluation_config.trained_model_dir, self.model_evaluation_config.trained_model_name)
            model = pickle.load(open(file_name, 'rb'))

            # Sample 100 items for evaluation
            sample = book_sparse[:100]

            start_time = time.time()
            distances, indices = model.kneighbors(sample, n_neighbors=6)
            elapsed_time = time.time() - start_time

            logging.info(f"Model evaluation: NearestNeighbors kneighbors on 100 samples took {elapsed_time:.4f} seconds")
            for i in range(5):
                logging.info(f"Sample {i} neighbors indices: {indices[i]}")
                logging.info(f"Sample {i} neighbors distances: {distances[i]}")

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_model_evaluation(self):
        try:
            logging.info(f"{'='*20}Model Evaluation log started.{'='*20} ")
            self.evaluate()
            logging.info(f"{'='*20}Model Evaluation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e
