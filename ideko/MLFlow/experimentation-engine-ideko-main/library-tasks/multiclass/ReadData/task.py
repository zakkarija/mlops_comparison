[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import pandas as pd
from classes import preprocessing_functions
from helpers.logger import LoggerHelper, logging


LoggerHelper.init_logger()
logger = logging.getLogger(__name__)

input_data_folder = variables.get("FileToRead")

indicator_list = ["f3"]
X, Y = preprocessing_functions.read_data(input_data_folder, indicator_list)
logger.info("Summary of timeseries length %s" %pd.Series([len(x) for x in X]).describe())

ph.save_datasets(variables, ("X41", X), ("Y", Y), ("IndicatorList", indicator_list))
