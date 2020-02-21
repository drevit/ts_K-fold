import itertools
import os
from globals import *
import pandas as pd
from tensorflow.keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.cm as cm


# =============================================================================
# RMSE : ROOT MEAN SQUARE ERROR
# =============================================================================

def RMSE(y_true, y_pred):
	import numpy as np

	if len(y_true) != len(y_pred):
		print('dimensioni di y_true e y_pred non consistenti')

	season = len(y_true)
	se = np.square(y_true - y_pred)
	rmse = np.sqrt((1 / season) * np.sum(se))

	return rmse

# =============================================================================
# MAPE : MEAN ABSOLUTE PERCENTAGE ERROR
# =============================================================================

def MAPE(y_true, y_pred):
	import numpy as np

	if len(y_true) != len(y_pred):
		print('dimensioni di y_true e y_pred non consistenti')

	ae = np.absolute(y_true - y_pred)
	ape = np.absolute(ae / y_true)
	mape = np.mean(ape)

	return mape




def get_unique_configs(det_layers1, det_hidden_sizes1):

	assert len(det_layers1) == len(det_hidden_sizes1)

	hyperparameters_list = []

	configs = []

	for lay1, hid1 in zip(det_layers1, det_hidden_sizes1):
		configs.append([lay1,hid1])

	configs_t = tuple(configs)
	unique_configs = []

	for counter, value in enumerate(configs_t):
		temp_config = list(configs_t)
		temp_config.pop(counter)
		if (value in temp_config) and (value in unique_configs):
			pass
		else:
			unique_configs.append(value)

	unique_configs = tuple(unique_configs)

	print('-----------------------------------------------------------------------')
	print('The following configurations will be evaluated:')
	for config in unique_configs:
		print(str(config))
	print('-----------------------------------------------------------------------')
	return unique_configs

def evaluate_config_metrics(path_run):

	train_mse_list = []
	train_mae_list = []
	train_mape_list = []

	vali_mse_list = []
	vali_mae_list = []
	vali_mape_list = []

	test_mse_list = []
	test_mae_list = []
	test_mape_list = []

	for hour in range(24):
		df_metrics = pd.read_csv(os.path.join(path_run, 'H_' + str(hour), 'metrics.csv'))

		train_mse_list.append(df_metrics['MSE'][df_metrics['#'] == 'TRAIN_AVG'].values)
		train_mae_list.append(df_metrics['MAE'][df_metrics['#'] == 'TRAIN_AVG'].values)
		train_mape_list.append(df_metrics['MAPE'][df_metrics['#'] == 'TRAIN_AVG'].values)

		vali_mse_list.append(df_metrics['MSE'][df_metrics['#'] == 'VALI_AVG'].values)
		vali_mae_list.append(df_metrics['MAE'][df_metrics['#'] == 'VALI_AVG'].values)
		vali_mape_list.append(df_metrics['MAPE'][df_metrics['#'] == 'VALI_AVG'].values)

		test_mse_list.append(df_metrics['MSE'][df_metrics['#'] == 'TEST'].values)
		test_mae_list.append(df_metrics['MAE'][df_metrics['#'] == 'TEST'].values)
		test_mape_list.append(df_metrics['MAPE'][df_metrics['#'] == 'TEST'].values)

	with open(os.path.join(path_run, 'metrics.csv'),'w') as f:
		buffer = ','.join(['#','MSE','MAE','MAPE']) + '\n' + \
				 ','.join(['TRAIN_AVG', str(np.mean(train_mse_list)), str(np.mean(train_mae_list)), str(np.mean(train_mape_list))]) + '\n' + \
				 ','.join(['VALI_AVG', str(np.mean(vali_mse_list)), str(np.mean(vali_mae_list)), str(np.mean(vali_mape_list))]) + '\n' + \
				 ','.join(['TEST_AVG', str(np.mean(test_mse_list)), str(np.mean(test_mae_list)), str(np.mean(test_mape_list))])
		f.write(buffer)
		f.close()


# TODO: evaluate hourly metrics