'''
Script for model training
'''

import os
import sys
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers import config
from helpers import logger
import numpy as np
import pandas as pd
from classes import preprocessing_functions
from classes.multiclass_models import NeuralNetwork, ConvolutionalNeuralNetwork, RecurrentNeuralNetwork, LongShortTermMemory

# Load logger & config
LoggerHelper.init_logger()
logger = logging.getLogger(__name__)
config = ConfigHelper.instance("models")

############################################################################################################
# DATA PREPROCESSING
############################################################################################################

# Define paths
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
os.makedirs(output_path, exist_ok = True)

# List of indicators to be read from the file
indicator_list = ["f3"]

################################
# READ DATA
################################

X, Y = preprocessing_functions.read_data(data_path, indicator_list)
logger.info("Summary of timeseries length %s" %pd.Series([len(x) for x in X]).describe())

# NOTE: The length of all the timeseries in X is not the same.
# Each file has a different number of points.

################################
# PADDING
################################

# To equal the length of the signals add padding (add zeros or nans at the end of the series).

X_pad = preprocessing_functions.add_padding(X, indicator_list)

Y_encoded = preprocessing_functions.encode_response_variable(Y)

logger.info("SHAPE X (%d, %d, %d)" %(np.shape(X_pad)))

logger.info("Number of timeseries %d" %np.shape(X_pad)[0])
logger.info("Number of points %d" %np.shape(X_pad)[1])
logger.info("Number of features %d" %np.shape(X_pad)[2])

logger.info("SHAPE X (%d, %d)" %(np.shape(Y_encoded)))

logger.info("Number of timeseries %d" %np.shape(Y_encoded)[0])
logger.info("Number of classes %d" %np.shape(Y_encoded)[1])


################################
# SPLIT DATA
################################

# Split data into training set and test set

X_train, X_test, y_train, y_test = preprocessing_functions.split_data(X_pad, Y_encoded)

n_timestamps = X_train.shape[1]
n_features = X_train.shape[2]

n_classes = y_train.shape[1]


############################################################################################################
# MODEL TRAINING
############################################################################################################

################################
# NEURAL NETWORK
################################

# Read NeuralNetwork config
config_nn = config["NeuralNetwork"]

# Boolean that indicates whether to train this model or not
enabled_nn = config_nn["enabled"]

if enabled_nn:

    logger.info("NEURAL NETWORK")

    # Model name
    folder_name = config_nn["name_parameters"]["folder_name"]
    model_name = config_nn["name_parameters"]["model_name"]

    # Path to store the model
    output_path_nn = os.path.join(output_path, folder_name)
    os.makedirs(output_path_nn, exist_ok = True)

    # Parameters defining the architecture of the model

    # Activation function
    activation_function = config_nn["model_parameters"]["activation_function"]

    # The length of the list is the number of layers and each element indicates the number of neurons in each layer.
    units = config_nn["model_parameters"]["units"]

    # Number of epochs y batch_size
    epochs = config_nn["training_parameters"]["epochs"]
    batch_size = config_nn["training_parameters"]["batch_size"]

    # Initialise the model class
    model_nn = NeuralNetwork(n_timestamps, n_features, activation_function, units, n_classes)

    # Create the model (according to the architecture defined)
    model_nn.create_model()

    # Define callbacks
    early_stopping = model_nn.early_stopping_callback()
    model_checkpoint = model_nn.model_checkpoint_callback(model_path = os.path.join(output_path_nn, model_name))
    callback_list = [early_stopping, model_checkpoint]

    # Model configuration and model training
    model_nn.model_compilation(model_nn.model)
    history_nn = model_nn.model_fitting(model_nn.model, X_train, y_train, X_test, y_test, callback_list, epochs, batch_size)

    # Plot history of the model
    preprocessing_functions.plot_model_history(history_nn, output_path_nn)

    # Model evaluation
    model_nn.model_evaluation(model_nn.model, X_pad, Y_encoded, X_test, y_test)

    # Metrics whole dataset
    logger.info("Metrics WHOLE DATASET")
    model_nn.compute_metrics(model_nn.model, X_pad, Y_encoded)

    # Metrics test data
    logger.info("Metrics TEST DATA")
    model_nn.compute_metrics(model_nn.model, X_test, y_test)


########################################
# CONVOLUTIONAL NEURAL NETWORK (CNN)
########################################

# Read CNN config
config_cnn = config["CNN"]

# Boolean that indicates whether to train this model or not
enabled_cnn = config_cnn["enabled"]

if enabled_cnn:

    logger.info("CONVOLUTIONAL NEURAL NETWORK (CNN)")

    # Model name
    folder_name = config_cnn["name_parameters"]["folder_name"]
    model_name = config_cnn["name_parameters"]["model_name"]

    # Path to store the model
    output_path_cnn = os.path.join(output_path, folder_name)
    os.makedirs(output_path_cnn, exist_ok = True)

    # Parameters defining the architecture of the model

    # Activation function
    activation_function = config_cnn["model_parameters"]["activation_function"]

    # The length of the list is the number of layers and each element indicates the number of filters.
    filters = config_cnn["model_parameters"]["filters"]

    # Filter and pooling size
    kernel_size = config_cnn["model_parameters"]["kernel_size"]
    pool_size = config_cnn["model_parameters"]["pool_size"]

    # Number of epochs and batch_size
    epochs = config_cnn["training_parameters"]["epochs"]
    batch_size = config_cnn["training_parameters"]["batch_size"]

    # Initialise the model class
    model_cnn = ConvolutionalNeuralNetwork(n_timestamps, n_features, activation_function, filters, kernel_size, pool_size, n_classes)

    # Create the model (according to the architecture defined)
    model_cnn.create_model()

    # Define callbacks
    early_stopping = model_cnn.early_stopping_callback()
    model_checkpoint = model_cnn.model_checkpoint_callback(model_path = os.path.join(output_path_cnn, model_name))
    callback_list = [early_stopping, model_checkpoint]

    # Model configuration and model training
    model_cnn.model_compilation(model_cnn.model)
    history_cnn = model_cnn.model_fitting(model_cnn.model, X_train, y_train, X_test, y_test, callback_list, epochs, batch_size)

    # Plote history of the model
    preprocessing_functions.plot_model_history(history_cnn, output_path_cnn)

    # Model evaluation
    model_cnn.model_evaluation(model_cnn.model, X_pad, Y_encoded, X_test, y_test)

    # Metrics whole dataset
    logger.info("Metrics WHOLE DATASET")
    model_cnn.compute_metrics(model_cnn.model, X_pad, Y_encoded)

    # Metrics test data
    logger.info("Metrics TEST DATA")
    model_cnn.compute_metrics(model_cnn.model, X_test, y_test)


################################
# RECURRENT NEURAL NETWORK (RNN)
################################

# Leer RNN config
config_rnn = config["RNN"]

# Boolean that indicates whether to train this model or not.
enabled_rnn = config_rnn["enabled"]

if enabled_rnn:

    logger.info("RECURRENT NEURAL NETWORK (RNN)")

    # Model name
    folder_name = config_rnn["name_parameters"]["folder_name"]
    model_name = config_rnn["name_parameters"]["model_name"]

    # Path to store the model
    output_path_rnn = os.path.join(output_path, folder_name)
    os.makedirs(output_path_rnn, exist_ok = True)

    # Parameters defining the architecture of the model

    # Activation function
    activation_function = config_rnn["model_parameters"]["activation_function"]

    # Layers and neurons in each layer
    hidden_units = config_rnn["model_parameters"]["hidden_units"]

    # Number of epochs and batch_size
    epochs = config_rnn["training_parameters"]["epochs"]
    batch_size = config_rnn["training_parameters"]["batch_size"]

    # Initialise the model class
    model_rnn = RecurrentNeuralNetwork(n_timestamps, n_features, activation_function, hidden_units, n_classes)

    # Create the model (according to the architecture defined)
    model_rnn.create_model()

    # Define callbacks
    early_stopping = model_rnn.early_stopping_callback()
    model_checkpoint = model_rnn.model_checkpoint_callback(model_path = os.path.join(output_path_rnn, model_name))
    callback_list = [early_stopping, model_checkpoint]

    # Model configuration and training
    model_rnn.model_compilation(model_rnn.model)
    history_rnn = model_rnn.model_fitting(model_rnn.model, X_train, y_train, X_test, y_test, callback_list, epochs, batch_size)

    # Plot history of the model
    preprocessing_functions.plot_model_history(history_rnn, output_path_rnn)

    # Model evaluation
    model_rnn.model_evaluation(model_rnn.model, X_pad, Y_encoded, X_test, y_test)

    # Metrics whole dataset
    logger.info("Metrics WHOLE DATASET")
    model_rnn.compute_metrics(model_rnn.model, X_pad, Y_encoded)

    # Metrics test data
    logger.info("Metrics TEST DATA")
    model_rnn.compute_metrics(model_rnn.model, X_test, y_test)


################################
# LONG SHORT TERM MEMORY (LSTM)
################################

# Read LSTM config
config_lstm = config["LSTM"]

# Boolean that indicates whether to train this model or not
enabled_lstm = config_lstm["enabled"]

if enabled_lstm:

    logger.info("LONG SHORT TERM MEMORY (LSTM)")

    # Model name
    folder_name = config_lstm["name_parameters"]["folder_name"]
    model_name = config_lstm["name_parameters"]["model_name"]

    # Path to store the model
    output_path_lstm = os.path.join(output_path, folder_name)
    os.makedirs(output_path_lstm, exist_ok = True)

    # Parameters defining the architecture of the model

    # Activation function
    activation_function = config_lstm["model_parameters"]["activation_function"]

    # Number of layers and the dimensions
    hidden_units = config_lstm["model_parameters"]["hidden_units"]

    # Number of epochs and batch_size
    epochs = config_lstm["training_parameters"]["epochs"]
    batch_size = config_lstm["training_parameters"]["batch_size"]

    # Initialise the model class
    model_lstm = LongShortTermMemory(n_timestamps, n_features, activation_function, hidden_units, n_classes)

    # Create the model (according to the architecture defined)
    model_lstm.create_model()

    # Definir callbacks
    early_stopping = model_lstm.early_stopping_callback()
    model_checkpoint = model_lstm.model_checkpoint_callback(model_path = os.path.join(output_path_lstm, model_name))
    callback_list = [early_stopping, model_checkpoint]

    # Model configuration and training
    model_lstm.model_compilation(model_lstm.model)
    history_lstm = model_lstm.model_fitting(model_lstm.model, X_train, y_train, X_test, y_test, callback_list, epochs, batch_size)

    # Plot history of the model
    preprocessing_functions.plot_model_history(history_lstm, output_path_lstm)

    # Model evaluation
    model_lstm.model_evaluation(model_lstm.model, X_pad, Y_encoded, X_test, y_test)

    # Metrics whole dataset
    logger.info("Metrics WHOLE DATASET")
    model_lstm.compute_metrics(model_lstm.model, X_pad, Y_encoded)

    # Metrics test data
    logger.info("Metrics TEST DATA")
    model_lstm.compute_metrics(model_lstm.model, X_test, y_test)
