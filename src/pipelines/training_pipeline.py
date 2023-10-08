import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    data_transfromation = DataTransformation()
    data = data_transfromation.initiate_data_transformation(train_data_path, test_data_path)
    print(len(data))

    if len(data) == 2:
        train_arr, test_arr = data
        print("brc")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)