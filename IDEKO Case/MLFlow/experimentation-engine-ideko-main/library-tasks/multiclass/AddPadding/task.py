[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import numpy as np
import os
import json
from classes import preprocessing_functions
from helpers.logger import LoggerHelper, logging

LoggerHelper.init_logger()
logger = logging.getLogger(__name__)

X, Y, indicator_list = ph.load_datasets(variables, "X42", "Y", "IndicatorList")

X_pad = preprocessing_functions.add_padding(X, indicator_list)

# TODO changed according to IDEKO mutliclass code
#Y_pad = np.array(Y)
Y_encoded = preprocessing_functions.encode_response_variable(Y)

logger.info("SHAPE X (%d, %d, %d)" %(np.shape(X_pad)))

logger.info("Number of timeseries %d" %np.shape(X_pad)[0])
logger.info("Number of points %d" %np.shape(X_pad)[1])
logger.info("Number of features %d" %np.shape(X_pad)[2])

# logger.info("SHAPE Y %s" %np.shape(Y_pad))
logger.info("SHAPE Y (%d, %d)" %(np.shape(Y_encoded)))

# ph.save_datasets(variables, ("XPad", X_pad), ("YPad", json.dumps(Y_pad,cls=ph.NumpyEncoder)))
ph.save_datasets(variables, ("XPad", X_pad), ("YPad", json.dumps(Y_encoded, cls=ph.NumpyEncoder)))
