import os
import datetime
from globals import *
import matplotlib.pyplot as plt

from others import *
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


import matplotlib as mpl
import matplotlib.cm as cm

class mdn4epf(Sequential):
	def __init__(self, det_layers, det_hidden_sizes, features_names,train_data, vali_data, device='GPU:0'):
		'''

		det_layers:       list of strings. Type of layers and their order in Sequential.
		                  Available values: MLP, RNN, GRU, LSTM

		det_hidden_sizes: list of ints. Hidden sizes of the layers in det_layers.

		n_mixes:          int.number of components of the Mixture Density

		'''

		print('Compiling model...')
		# clear any pre-existing tensorflow graph
		clear_session()

		super(mdn4epf, self).__init__()

		# build and compile the graph

		if type(det_layers) != list and type(det_hidden_sizes) != list:
			det_layers = [det_layers]
			det_hidden_sizes = [det_hidden_sizes]

		if INCLUDE_L1_REG == True:
			print('L1 regularizer applied to the first deterministic layer')
			reg = 'l1'
		else:
			print('Not regularizing weights and biases')
			reg = None
		flag = True

		for lay_str, hidden_size in zip(det_layers, det_hidden_sizes):

			# if L1 regularization is expected, we apply it only on the first deterministic layer
			if flag == True:
				pass
			else:
				reg = None

			flag = False

			if lay_str == 'MLP':
				self.add(Dense(hidden_size, activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg))
			elif lay_str == 'RNN':
				self.add(SimpleRNN(hidden_size, activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg))
			elif lay_str == 'GRU':
				self.add(GRU(hidden_size, activation = 'relu', kernel_regularizer=reg, bias_regularizer=reg))
			elif lay_str == 'LSTM':
				self.add(LSTM(hidden_size, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
			else:
				raise NotImplementedError

		self.add(Dense(1, activation='linear'))

		self.compile(loss='mse', optimizer=Adam(), metrics=['mse','mae','mape'])

		tf.device(device)



		# initialize best_mse value and lists to evaluate overall (not fold-specific) train and vali performance
		self.fold_train_mse = []
		self.fold_train_mae = []
		self.fold_train_mape = []

		self.best_mse = 10000000

		self.fold_vali_mse = []
		self.fold_vali_mae = []
		self.fold_vali_mape = []

		self.features_names = features_names

		self.X_train, self.y_train_ref = train_data
		self.X_vali, self.y_vali_ref = vali_data

		print('Done.')

	def setup_folders(self, det_layers, det_hidden_sizes, hour, fold_n):
		print('=========================================================================')
		# crate directories to save checkpoints and tensorboard log

		self.hour = hour
		self.fold_n = fold_n

		if type(det_layers) != list and type(det_hidden_sizes) != list:
			det_layers = [det_layers]
			det_hidden_sizes = [det_hidden_sizes]

		self.folder_run_name = datetime.datetime.now().strftime(format='%Y%m%d') + '_' + '_'.join([str(lay)+str(size) for lay,size in zip(det_layers, det_hidden_sizes)])
		self.path_run = os.path.join(DIR_RUNS, self.folder_run_name)
		if self.folder_run_name not in os.listdir(DIR_RUNS):
			print('Creating run directory: ' + self.path_run)
			os.mkdir(self.path_run)
		else:
			print('Using existing run directory: ' + self.path_run)

		self.folder_hour_name = 'H_' + str(self.hour)
		self.path_hour = os.path.join(DIR_RUNS, self.folder_run_name, self.folder_hour_name)
		if self.folder_hour_name not in os.listdir(self.path_run):
			print('Creating hour directory: ' + self.path_hour)
			os.mkdir(self.path_hour)
		else:
			print('Using existing hour directory: ' + self.path_hour)

		# create csv for metrics
		if 'metrics.csv' not in os.listdir(self.path_hour):
			filename = os.path.join(self.path_hour, 'metrics.csv')
			with open(filename, 'a') as f:
				f.write(','.join(['#','MSE','MAE','MAPE']) + '\n')
				f.close()


		self.folder_fold_name = 'FOLD_'+ str(fold_n)
		print('Setting up fold ' + str(fold_n) + ' ...')

		assert self.folder_fold_name not in os.listdir(self.path_hour)
		self.path_fold = os.path.join(self.path_hour, self.folder_fold_name)
		os.mkdir(self.path_fold)

		# callbacks
		# setting ModelCheckpoint parameter save_best_only=True saves only the best performing model
		checkpointBest = ModelCheckpoint(filepath=os.path.join(self.path_fold,'checkpoint'),save_best_only=True) #' os.path.join(self.path_hour, self.fold_name)
		checkpoint10 = ModelCheckpoint(filepath = os.path.join(self.path_fold, 'checkpoint_ep{epoch:02d}'), save_best_only=False, period=10)
		csvlogger = CSVLogger(os.path.join(self.path_fold,'training_log.csv'))
		# setting TensorBoard parameter write_graph=True enable graph visualization but log file increases in size
		tensorboard = TensorBoard(log_dir=self.path_fold, write_graph=False)
		earlystopping = EarlyStopping(patience=PATIENCE,restore_best_weights=True,min_delta=0.1)


		self.call_list = [tensorboard, earlystopping, checkpointBest, checkpoint10, csvlogger]

		print('Done.')

	def evaluate_fold_vali_metrics(self, fold_n):
		print('Validating current fold...')

		self.y_train_pred= self.predict(self.X_train)
		self.y_vali_pred= self.predict(self.X_vali)

		self.mse_train = MSE(y_true=self.y_train_ref, y_pred=self.y_train_pred)
		self.mse_vali = MSE(y_true=self.y_vali_ref, y_pred=self.y_vali_pred)

		self.mae_train = MAE(y_true=self.y_train_ref, y_pred=self.y_train_pred)
		self.mae_vali = MAE(y_true=self.y_vali_ref, y_pred=self.y_vali_pred)

		self.mape_train = MAPE(y_true=self.y_train_ref, y_pred=self.y_train_pred)
		self.mape_vali = MAPE(y_true=self.y_vali_ref, y_pred=self.y_vali_pred)



		assert fold_n in FOLDS_VALI

		# update metrics_MaxLike.csv
		filename = os.path.join(self.path_hour, 'metrics.csv')

		with open(filename, 'a') as f:
			f.write(','.join(['TRAIN_F' + str(fold_n), str(self.mse_train), str(self.mae_train), str(self.mape_train)]) + '\n')

			if self.mse_vali < self.best_mse:
				self.best_mse = self.mse_vali
				self.best_fold = self.folder_fold_name

			f.write(','.join(['VALI_F' + str(fold_n), str(self.mse_vali), str(self.mae_vali), str(self.mape_vali)]) + '\n')
			f.close()


	def evaluate_avg_vali_metrics(self,fold_train_mse,fold_train_mae,fold_train_mape, fold_vali_mse,fold_vali_mae,fold_vali_mape):

		# update metrics.csv
		filename = os.path.join(self.path_hour, 'metrics.csv')

		with open(filename, 'a') as f:
			f.write(','.join(['TRAIN_AVG', str(np.mean(fold_train_mse)), str(np.mean(fold_train_mae)), str(np.mean(fold_train_mape))]) + '\n')
			f.write(','.join(['VALI_AVG', str(np.mean(fold_vali_mse)), str(np.mean(fold_vali_mae)), str(np.mean(fold_vali_mape))]) + '\n')
			f.close()

	def evaluate_test_set(self, inputs, targets):

		# the test set is divided in 6 folds like the vali_set. First of all we merge the test folds
		self.X_test = np.vstack([inputs['vali']['H_' + str(self.hour)][str(fold_n)] for fold_n in FOLDS_TEST])
		self.y_test_ref = np.vstack([targets['vali']['H_' + str(self.hour)][str(fold_n)] for fold_n in FOLDS_TEST])

		#then we retrieve the model that performed best in the k-fold crossvalidation phase
		self.load_weights(os.path.join(self.path_hour, self.best_fold, 'checkpoint'))

		# obtain mu, sig, pis predictions (or point prediction in case of full deterministic net)

		self.y_test_pred = self.predict(self.X_test)

		self.test_mse = MSE(y_true=self.y_test_ref, y_pred=self.y_test_pred)
		self.test_mae = MAE(y_true=self.y_test_ref, y_pred=self.y_test_pred)
		self.test_mape = MAPE(y_true=self.y_test_ref, y_pred=self.y_test_pred)

		# update metrics.csv
		filename = os.path.join(self.path_hour, 'metrics.csv')

		with open(filename, 'a') as f:
			f.write(','.join(['TEST', str(self.test_mse), str(self.test_mae), str(self.test_mape)]) + '\n')
			f.close()


