[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import os
import numpy as np

# from classes.multiclass_models import NeuralNetwork

from classes import model_configuration
from classes import preprocessing_functions
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper

logger = logging.getLogger(__name__)

metrics_translation = {
    "accuracy": "IDEKO_accuracy",
    "recall": "IDEKO_recall",
    "loss": "IDEKO_loss"
}

def model_evaluation(model, X, y, X_test, y_test, myvariables, myresultMap):
    ''' Function to evaluate the model on the entire dataset and on the test set.

    Parameters (input):
        model (model)
        X (tensor): dataset with structure (n_series, n_timestamps, n_features)
        y (array): array of the labels of the dataset
        X_test (tensor): test data
        y_test (array): array of the labels of test data
    '''

    logger.info("model_evaluation(): Evaluation of the model with metrics %s" %model.metrics_names)

    model_metrics_data = model.evaluate(X, y, return_dict = True)
    for metric, value in model_metrics_data.items():
        logger.info("model_evaluation(): %s of the dataset: %s" %(metric, value))

    model_metrics_test = model.evaluate(X_test, y_test, return_dict = True)
    for metric, value in model_metrics_test.items():
        myresultMap.put(metrics_translation[metric], value)
        logger.info("model_evaluation(): %s of the test data: %s" %(metric, value))

    return myresultMap

n_timestamps, n_features, n_classes = ph.load_datasets(variables, "Timestamps2", "Features", "n_classes")
X_test, y_test = ph.load_datasets(variables, "XTest", "YTest")
X_pad, Y_pad = ph.load_datasets(variables, "XPad", "YPad")
model_path = ph.load_datasets(variables, "OutputFolder")

model = keras.models.load_model(model_path)

y_test = np.asarray(y_test)
Y_pad = np.asarray(Y_pad)
resultMap = model_evaluation(model, X_pad, Y_pad, X_test, y_test, variables, resultMap)

ph.save_datasets(variables, ("OutputFolder", model_path))
