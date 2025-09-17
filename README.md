# Book Recommendation System

## Overview

This project implements a Book Recommendation System using collaborative filtering based on a Nearest Neighbors clustering model. The system provides personalized book recommendations by analyzing users’ rating patterns and identifying similar books.

## Features

- Collaborative filtering using user-item rating data.
- Nearest Neighbors model built with brute-force similarity search.
- Data ingestion, validation, transformation, model training, and evaluation stages modularized in pipeline components.
- Efficient pivot table creation for user-item ratings.
- Model evaluation focusing on nearest neighbor query latency and result review.
- All processes logged for easy debugging and monitoring.
- Deployed as a production-ready service on an AWS EC2 instance for scalable, cloud-based access.

## Architecture

The recommendation engine workflow consists of:

1. **Data Ingestion**: Download book rating datasets and extract the data.
2. **Data Validation**: Clean and preprocess the rating data, keeping users and books with sufficient interactions.
3. **Data Transformation**: Transform cleaned data to create a pivot matrix of books vs users.
4. **Model Training**: Train a Nearest Neighbors unsupervised model on the user-book rating pivot.
5. **Model Evaluation**: Evaluate the trained model using latency and sample nearest neighbor inspections.
6. **Recommendation Service**: (Deployed on AWS EC2) Exposes the trained model for real-time book recommendations.

## Technologies Used

- Pandas, Scipy, scikit-learn
- Collaborative Filtering with NearestNeighbors algorithm
- Logging for monitoring and debugging
- AWS EC2 for deployment


## How to Use

1. Clone the repository.
2. Install dependencies (e.g., `pip install -r requirements.txt`).
3. Configure application settings in `configuration.py` for data URLs and directories.
4. Run the training pipeline to build and evaluate the model.


## Evaluation

Model evaluation is tailored to nearest neighbor search — focusing on:

- Query execution time for nearest neighbors.
- Inspection of neighbor indices and distances to verify recommendation relevance.
- No classical supervised metrics (accuracy, precision) as this is an unsupervised similarity-based model.

## Future Improvements

- Incorporate content-based features for hybrid recommendations.
- Add user feedback loop for improving recommendation relevance.
- Implement scalable approximate nearest neighbor search for faster queries.
- Build a user-friendly front-end interface.



