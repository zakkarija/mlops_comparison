[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import os
import json
import numpy as np
from classes import preprocessing_functions


# X_pad, Y_pad_json = ph.load_datasets(variables, "XPad", "YPad")
# Y_pad = json.loads(Y_pad_json)

X_pad, Y_pad_json = ph.load_datasets(variables, "XPad", "YPad")
Y_pad = json.loads(Y_pad_json)

# Split data into training set and test set

X_train, X_test, y_train, y_test = preprocessing_functions.split_data(X_pad, Y_pad)

n_timestamps = X_train.shape[1]
n_features = X_train.shape[2]

n_classes = np.shape(y_train)[1]
# n_classes = y_train.shape[1]

ph.save_datasets(variables, ("Timestamps41", n_timestamps), ("Features", n_features), ("n_classes", n_classes))
ph.save_datasets(variables, ("XTrain", X_train), ("XTest", X_test))
ph.save_datasets(variables, ("YTrain", y_train), ("YTest", y_test))
ph.save_datasets(variables, ("XPad", X_pad), ("YPad", Y_pad))
