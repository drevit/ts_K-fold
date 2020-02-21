import os
DIR_MAIN = os.path.dirname(os.path.realpath(__file__))
DIR_DATA = os.path.join(DIR_MAIN, 'data')
DIR_RUNS = os.path.join(DIR_MAIN, 'runs')
RAW_XLSX_FILENAME = 'data.xlsx'


SCALER_TYPE = 'robust'
IS_MODEL_RECURRENT = True
ENABLE_PLOTS = False # set to False to consistently spped up training
NORMALIZE_TARGET = False
INCLUDE_L1_REG = True

DET_LAYER_1 = [['GRU'], ['RNN']]
DET_HIDDEN_SIZE_1 = [[20], [50]]




BATCH_SIZE = 64  # default 64
MAX_EPOCHS = 3   # default 1000
PATIENCE = 2     # default 30

FOLDS_VALI = [1,2,4,5,6] # fold 3 is missing due to lack of data
FOLDS_TEST = [7,8,9,10,11,12]
