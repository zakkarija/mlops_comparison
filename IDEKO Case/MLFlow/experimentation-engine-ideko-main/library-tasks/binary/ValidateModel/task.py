[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph

import os
import zipfile
import numpy as np
import pandas as pd

import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper

logger = logging.getLogger(__name__)

logger.info("TASK VALIDATE MODEL")

def post_result(file, model_result, myresultMap):

    if model_result == 0:
        logger.info("post_result(): The model has classified the data as not anomalous")
        myresultMap.put(file, "not_anomalous")

    elif model_result == 1:
        logger.info("post_result(): The model has classified the data as anomalous")
        myresultMap.put(file, "anomalous")

    return myresultMap

# Load the trained model
model_path = ph.load_datasets(variables, "OutputFolder")
model = keras.models.load_model(model_path)

# Number of timestamps and features of the model
n_timestamps_model = model.input_shape[1]
n_features_model = model.input_shape[2]

# Read data
input_data_folder = variables.get("FileToValidate")
indicator_list = ["f3"]

labels = {
        "not_anomalous": 0,
        "mechanical_anomalies": 1,
        "electrical_anomalies": 1
    }

threshold = 0.5

# Folder list
folders = os.listdir(input_data_folder)

# Loop for the files in the folder
for folder in labels.keys():
    folder_path = os.path.join(input_data_folder, folder)
    files = os.listdir(folder_path)

    # The label of the data is the name of the folder (MECHANICAL ANOMALIES / NOT ANOMALOUS)
    y = labels[folder]

    # Loop for the files in the folder
    for file in files:

        # File path
        file_path = os.path.join(folder_path, file)

        # Unzip the file y read it as DataFrame
        zip_file = zipfile.ZipFile(file_path, 'r')
        filename = zip_file.namelist()[0]
        df = pd.read_csv(zip_file.open(filename), delimiter = ";")

        # Select the indicators
        X = df[indicator_list].to_numpy()

        # Reshape the data to the model input shape
        timestamps, features = X.shape
        reshaped_X = X.reshape(1, timestamps, features)

        # Pad the data to the model input shape
        padded_X = pad_sequences(reshaped_X, maxlen = n_timestamps_model, padding = "post", dtype = "float64", value = np.full((n_features_model,), 0))

        # Returns the probability that each element belongs to class 1.
        prob = model.predict(padded_X)

        # Calculate the binary array. Probabilities greater than threshold are classified as class 1.
        prediction = (prob.flatten() >= threshold).astype(int)[0]

        logger.info("File %s, true label %d, predicted label %d" %(file, y, prediction))

        resultMap = post_result(file, prediction, resultMap)






