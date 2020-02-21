from globals import *
from tensorflow.keras import Sequential
from data.dataset import generate_data, load_data
from models import mdn4epf
from others import get_unique_configs, evaluate_config_metrics

def main():

	inputs, targets = load_data()

	#TODO: improve get_unique_configs
	UNIQUE_CONFIGS = get_unique_configs(DET_LAYER_1, DET_HIDDEN_SIZE_1)

	for i, config in enumerate(UNIQUE_CONFIGS):
		print('Evaluating configuration %i / %i' %(i, len(UNIQUE_CONFIGS)))

		for hour in range(24):

			# TODO: make it more elegant
			fold_train_mse = []
			fold_train_mae = []
			fold_train_mape = []

			fold_vali_mse = []
			fold_vali_mae = []
			fold_vali_mape = []


			for fold_n in FOLDS_VALI:
				#TODO: if IS_MODEL_RECURRENT == True, input data is a 3D array not shaped as a Dataframe. Better save
				# feature names in a separate .pkl and load it

				# features_names = inputs['train']['H_'+ str(hour)][str(fold_n)].columns.tolist()
				features_names = []

				# TODO: if IS_MODEL_RECURRENT == True, input data is a 3D array, so the .values attribute is not required. Make it more elegant

				if IS_MODEL_RECURRENT:

					X_train = inputs['train']['H_' + str(hour)][str(fold_n)]
					y_train = targets['train']['H_' + str(hour)][str(fold_n)].values

					X_vali = inputs['vali']['H_' + str(hour)][str(fold_n)]
					y_vali = targets['vali']['H_' + str(hour)][str(fold_n)].values

				else:

					X_train = inputs['train']['H_'+ str(hour)][str(fold_n)].values
					y_train = targets['train']['H_'+ str(hour)][str(fold_n)].values

					X_vali = inputs['vali']['H_'+ str(hour)][str(fold_n)].values
					y_vali = targets['vali']['H_'+ str(hour)][str(fold_n)].values


				# TODO better implementation of how model read configuration (dictionary?)
				epf_model = mdn4epf(det_layers=config[0], det_hidden_sizes=config[1],
									features_names=features_names, device = 'GPU:0',
									train_data=(X_train,y_train), vali_data=(X_vali,y_vali))
				epf_model.setup_folders(det_layers=config[0], det_hidden_sizes=config[1], hour=hour,fold_n=fold_n)

				print('Training...')
				epf_model.fit(x=X_train, y=y_train,
							  batch_size=BATCH_SIZE,
							  epochs=MAX_EPOCHS,
							  callbacks=epf_model.call_list,
							  validation_data=(X_vali,y_vali))

				epf_model.summary()

				epf_model.evaluate_fold_vali_metrics(fold_n=fold_n)

				fold_train_mse.append(epf_model.mse_train)
				fold_train_mae.append(epf_model.mae_train)
				fold_train_mape.append(epf_model.mape_train)

				fold_vali_mse.append(epf_model.mse_vali)
				fold_vali_mae.append(epf_model.mae_vali)
				fold_vali_mape.append(epf_model.mape_vali)


			print('Validating current hour...')
			epf_model.evaluate_avg_vali_metrics(fold_train_mse,
												fold_train_mae,
												fold_train_mape,
												fold_vali_mse,
												fold_vali_mae,
												fold_vali_mape)

			epf_model.evaluate_test_set(inputs,targets)
			print('Done.')

		#TODO: plot results

		print('Validating current configuration...')
		evaluate_config_metrics(epf_model.path_run)
		print('Done.')




if __name__ == '__main__':
	main()

