[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import os
import numpy as np
from classes.binary_models import RecurrentNeuralNetwork
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper

logger = logging.getLogger(__name__)

config = ConfigHelper.instance("models")
config_rnn = config["RNN"]

logger.info("RECURRENT NEURAL NETWORK (RNN)")

# Model name
folder_name = config_rnn["name_parameters"]["folder_name"]
model_name = config_rnn["name_parameters"]["model_name"]

# Path to store the model
working_dir = os.getcwd()
output_path = os.path.join(working_dir, "output")
output_path_rnn = os.path.join(output_path, folder_name)
os.makedirs(output_path_rnn, exist_ok = True)

# creating directory to save the trained model
model_path = ph.create_dir(variables, 'trained_model')
output_data_folder = variables.get("OutputFolder")
# Parameters defining the architecture of the model

# Activation function
# activation_function = config_rnn["model_parameters"]["activation_function"]
activation_function = variables.get("activation_function")

# Layers and neurons in each layer
# hidden_units = config_rnn["model_parameters"]["hidden_units"]
# TODO: casted to int
hidden_units =  [int(variables.get("hidden_units_1")), int(variables.get("hidden_units_2"))]

# Number of epochs and batch_size
epochs = int(variables.get("epochs"))
batch_size = int(variables.get("batch_size"))

n_timestamps, n_features = ph.load_datasets(variables, "Timestamps", "Features")
X_train, y_train = ph.load_datasets(variables, "XTrain", "YTrain")
X_test, y_test = ph.load_datasets(variables, "XTest", "YTest")
X_pad, Y_pad = ph.load_datasets(variables, "XPad", "YPad")

print(f"n_timestamps in train_rn task: {n_timestamps}")
print(f"n_features in train_rn task: {n_features}")

model_rnn = RecurrentNeuralNetwork(n_timestamps, n_features, activation_function, hidden_units)

# Create the model (according to the architecture defined)
model_rnn.create_model()

# Define callbacks
early_stopping = model_rnn.early_stopping_callback()
model_checkpoint = model_rnn.model_checkpoint_callback(model_path = os.path.join(output_path_rnn, model_name))
callback_list = [early_stopping, model_checkpoint]

# Model configuration and training
model_rnn.model_compilation(model_rnn.model)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
history_rnn = model_rnn.model_fitting(model_rnn.model, X_train, y_train, X_test, y_test, callback_list, epochs, batch_size)

# Plot history of the model
# preprocessing_functions.plot_model_history(history_rnn, output_path_rnn)

# Model evaluation
# Y_pad = np.asarray(Y_pad)
# resultMap = model_rnn.model_evaluation(model_rnn.model, X_pad, Y_pad, X_test, y_test, variables, resultMap)

# added following lines to save trained model
model = model_rnn.model
model_path = os.path.join(model_path, model_name)
model.save(model_path)
model.save(output_data_folder)

# added following lines to save intermediate data for the next task
ph.save_datasets(variables, ("OutputFolder", model_path))
ph.save_datasets(variables, ("XTest", X_test))
ph.save_datasets(variables, ("YTest", y_test))
ph.save_datasets(variables, ("XPad", X_pad), ("YPad", Y_pad))
ph.save_datasets(variables, ("Timestamps", n_timestamps), ("Features", n_features))
ph.save_datasets(variables, ("model_path", model_path))
