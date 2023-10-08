import os
import pandas as pd
import joblib  
from src.logger import logging
from src.exception import CustomException



class PredictionPipeline:

    def __init__(self):
        # Load the pre-trained model and scaler during object initialization
        self.model = joblib.load(os.path.join('artifacts', 'model.pkl'))
        self.scaler = joblib.load(os.path.join('artifacts', 'preprocessor.pkl'))

    def predict(self, features):
        try:
            # Scale the input features using the loaded scaler
            scaled_data = self.scaler.transform(features)

            # Make a prediction using the loaded model
            pred = self.model.predict(scaled_data)

            return pred

        except Exception as e:
            # Handle exceptions and log them
            logging.info("Exception occurred in Prediction pipeline: {}".format(str(e)))
            raise CustomException(str(e))


class CustomData:

    def __init__(self, carat: float, depth: float, table: float, x: float, y: float, z: float, cut: str, color: str, clarity: str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            # Create a dictionary from the input data
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity],
            }

            # Create a Pandas DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df

        except Exception as e:
            # Handle exceptions and log them
            logging.info("Exception at prediction pipeline in get data: {}".format(str(e)))
            raise CustomException(str(e))
