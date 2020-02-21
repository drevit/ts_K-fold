# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:36:37 2018

@author: vitali
"""

import os
from globals import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import RobustScaler, StandardScaler

def generate_data(scaler_type):
    '''
    scaler: string. Identifies the type of scaler to use. Available values: 'robust', 'none', 'standard'
    '''

    print('Generating dataset...')
    print('-----------------------------------------------------------------------')
    xlsx_filename = os.path.join(DIR_DATA, RAW_XLSX_FILENAME)
    xlsx_sheetname = 'Sheet1'
    target_col_name = 'TARGET'

    df = pd.read_excel(xlsx_filename, sheet_name = xlsx_sheetname, index_col=0,  parse_dates = True)
    print('\nFeatures in ./data/' + xlsx_filename + ':\n')
    print('\n'.join(col for col in df.columns if col != target_col_name))
    print('\nTargets in ./data/' + xlsx_filename + ':\n')
    print('\n'.join(col for col in df.columns if col == target_col_name))
    print('-----------------------------------------------------------------------')

    for col in df.columns:
        if not(NORMALIZE_TARGET) and col == target_col_name:
            continue
        else:
            if scaler_type == 'robust':
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            elif scaler_type == 'standard':
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            elif scaler_type == 'none':
                pass
            else:
                raise NotImplementedError

    old_features_columns = df.columns.drop(target_col_name)
    new_features_columns = []

    df_features = df[old_features_columns]
    df_targets = df[target_col_name]

    for hour in range(24):
        for col in old_features_columns:

            new_col = col + ' D h:' + str(hour)
            new_features_columns.append(new_col)


    index_date = df.index.strftime('%Y-%m-%d').tolist()
    index_hour = df.index.tolist()

    features_data = np.zeros(shape = (len(index_hour), len(new_features_columns)))
    targets_data = np.zeros(shape = (len(index_hour), 1))

    counter = 0

    for i in range(len(index_hour)):

        temp_df_features = df_features.loc[index_date[counter]]
        temp_targets = df_targets.loc[index_hour[counter]]

        temp_features = temp_df_features.values.flatten()

        features_data[counter, :] = temp_features
        targets_data[counter] = temp_targets
        counter += 1

    df_features = pd.DataFrame(data = features_data,
                               columns = new_features_columns,
                               index = index_hour)

    df_targets = pd.DataFrame(data = targets_data,
                              columns = [target_col_name],
                              index = index_hour)





    # 1 -> 6  = VALI
    # 6 -> 12 = TEST: ONLY inputs['vali'] / targets['vali] TO BE USED, AND ONLY IN EVAL MODE (NO TRAINING)

    inputs = {}
    targets = {}

    inputs['train'] = {}
    inputs['vali'] = {}
    targets['train'] = {}
    targets['vali'] = {}

    for hour in range(24):

        if IS_MODEL_RECURRENT == False:

            inputs['train']['H_'+str(hour)] = {}
            inputs['vali']['H_'+str(hour)] = {}
            targets['train']['H_'+str(hour)] = {}
            targets['vali']['H_'+str(hour)] = {}

            df_features_h = df_features[df_features.index.hour == hour]
            df_targets_h = df_targets[df_targets.index.hour == hour]

            inputs['train']['H_'+str(hour)]['1'] = df_features_h.loc['2015-01-06 00:00':'2016-10-31 23:00']
            inputs['vali']['H_'+str(hour)]['1'] = df_features_h.loc['2016-11-01 00:00':'2016-12-31 23:00']
            targets['train']['H_'+str(hour)]['1'] = df_targets_h.loc['2015-01-06 00:00':'2016-10-31 23:00']
            targets['vali']['H_'+str(hour)]['1'] = df_targets_h.loc['2016-11-01 00:00':'2016-12-31 23:00']

            inputs['train']['H_'+str(hour)]['2'] = df_features_h.loc['2015-01-06 00:00':'2016-12-31 23:00']
            inputs['vali']['H_'+str(hour)]['2'] = df_features_h.loc['2017-01-01 00:00':'2017-02-28 23:00']
            targets['train']['H_'+str(hour)]['2'] = df_targets_h.loc['2015-01-06 00:00':'2016-12-31 23:00']
            targets['vali']['H_'+str(hour)]['2'] = df_targets_h.loc['2017-01-01 00:00':'2017-02-28 23:00']

            inputs['train']['H_'+str(hour)]['3'] = df_features_h.loc['2015-01-06 00:00':'2017-02-28 23:00']
            inputs['vali']['H_'+str(hour)]['3'] = df_features_h.loc['2017-03-01 00:00':'2017-04-30 23:00']
            targets['train']['H_'+str(hour)]['3'] = df_targets_h.loc['2015-01-06 00:00':'2017-02-28 23:00']
            targets['vali']['H_'+str(hour)]['3'] = df_targets_h.loc['2017-03-01 00:00':'2017-04-30 23:00']

            inputs['train']['H_'+str(hour)]['4'] = df_features_h.loc['2015-01-06 00:00':'2017-04-30 23:00']
            inputs['vali']['H_'+str(hour)]['4'] = df_features_h.loc['2017-05-01 00:00':'2017-06-30 23:00']
            targets['train']['H_'+str(hour)]['4'] = df_targets_h.loc['2015-01-06 00:00':'2017-04-30 23:00']
            targets['vali']['H_'+str(hour)]['4'] = df_targets_h.loc['2017-05-01 00:00':'2017-06-30 23:00']

            inputs['train']['H_'+str(hour)]['5'] = df_features_h.loc['2015-01-06 00:00':'2017-06-30 23:00']
            inputs['vali']['H_'+str(hour)]['5'] = df_features_h.loc['2017-07-01 00:00':'2017-08-31 23:00']
            targets['train']['H_'+str(hour)]['5'] = df_targets_h.loc['2015-01-06 00:00':'2017-06-30 23:00']
            targets['vali']['H_'+str(hour)]['5'] = df_targets_h.loc['2017-07-01 00:00':'2017-08-31 23:00']

            inputs['train']['H_'+str(hour)]['6'] = df_features_h.loc['2015-01-06 00:00':'2017-08-31 23:00']
            inputs['vali']['H_'+str(hour)]['6'] = df_features_h.loc['2017-09-01 00:00':'2017-10-31 23:00']
            targets['train']['H_'+str(hour)]['6'] = df_targets_h.loc['2015-01-06 00:00':'2017-08-31 23:00']
            targets['vali']['H_'+str(hour)]['6'] = df_targets_h.loc['2017-09-01 00:00':'2017-10-31 23:00']

            # inputs['train']['H_'+str(hour)]['7'] = df_features_h.loc['2015-01-06 00:00':'2017-10-31 23:00']
            inputs['vali']['H_'+str(hour)]['7'] = df_features_h.loc['2017-11-01 00:00':'2017-12-31 23:00']
            # targets['train']['H_'+str(hour)]['7'] = df_targets_h.loc['2015-01-06 00:00':'2017-10-31 23:00']
            targets['vali']['H_'+str(hour)]['7'] = df_targets_h.loc['2017-11-01 00:00':'2017-12-31 23:00']

            # inputs['train']['H_'+str(hour)]['8'] = df_features_h.loc['2015-01-06 00:00':'2017-12-31 23:00']
            inputs['vali']['H_'+str(hour)]['8'] = df_features_h.loc['2018-01-01 00:00':'2018-02-28 23:00']
            # targets['train']['H_'+str(hour)]['8'] = df_targets_h.loc['2015-01-06 00:00':'2017-12-31 23:00']
            targets['vali']['H_'+str(hour)]['8'] = df_targets_h.loc['2018-01-01 00:00':'2018-02-28 23:00']

            # inputs['train']['H_'+str(hour)]['9'] = df_features_h.loc['2015-01-06 00:00':'2018-02-28 23:00']
            inputs['vali']['H_'+str(hour)]['9'] = df_features_h.loc['2018-03-01 00:00':'2018-04-30 23:00']
            # targets['train']['H_'+str(hour)]['9'] = df_targets_h.loc['2015-01-06 00:00':'2018-02-28 23:00']
            targets['vali']['H_'+str(hour)]['9'] = df_targets_h.loc['2018-03-01 00:00':'2018-04-30 23:00']

            # inputs['train']['H_'+str(hour)]['10'] = df_features_h.loc['2015-01-06 00:00':'2018-04-30 23:00']
            inputs['vali']['H_'+str(hour)]['10'] = df_features_h.loc['2018-05-01 00:00':'2018-06-30 23:00']
            # targets['train']['H_'+str(hour)]['10'] = df_targets_h.loc['2015-01-06 00:00':'2018-04-30 23:00']
            targets['vali']['H_'+str(hour)]['10'] = df_targets_h.loc['2018-05-01 00:00':'2018-06-30 23:00']

            # inputs['train']['H_'+str(hour)]['11'] = df_features_h.loc['2015-01-06 00:00':'2018-06-30 23:00']
            inputs['vali']['H_'+str(hour)]['11'] = df_features_h.loc['2018-07-01 00:00':'2018-08-31 23:00']
            # targets['train']['H_'+str(hour)]['11'] = df_targets_h.loc['2015-01-06 00:00':'2018-06-30 23:00']
            targets['vali']['H_'+str(hour)]['11'] = df_targets_h.loc['2018-07-01 00:00':'2018-08-31 23:00']

            # inputs['train']['H_'+str(hour)]['12'] = df_features_h.loc['2015-01-06 00:00':'2018-08-31 23:00']
            inputs['vali']['H_'+str(hour)]['12'] = df_features_h.loc['2018-09-01 00:00':]
            # targets['train']['H_'+str(hour)]['12'] = df_targets_h.loc['2015-01-06 00:00':'2018-08-31 23:00']
            targets['vali']['H_'+str(hour)]['12'] = df_targets_h.loc['2018-09-01 00:00':]

        else:

            inputs['train']['H_' + str(hour)] = {}
            inputs['vali']['H_' + str(hour)] = {}
            targets['train']['H_' + str(hour)] = {}
            targets['vali']['H_' + str(hour)] = {}

            df_features_h = df_features[df_features.index.hour == hour]
            df_targets_h = df_targets[df_targets.index.hour == hour]

            inputs['train']['H_' + str(hour)]['1'] = np.array([df_features_h.loc['2015-01-06 00:00':'2016-10-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2015-01-06 00:00':'2016-10-31 23:00'].shape[0])])
            inputs['vali']['H_' + str(hour)]['1'] = np.array([df_features_h.loc['2016-11-01 00:00':'2016-12-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2016-11-01 00:00':'2016-12-31 23:00'].shape[0])])
            targets['train']['H_' + str(hour)]['1'] = df_targets_h.loc['2015-01-06 00:00':'2016-10-31 23:00']
            targets['vali']['H_' + str(hour)]['1'] = df_targets_h.loc['2016-11-01 00:00':'2016-12-31 23:00']

            inputs['train']['H_' + str(hour)]['2'] = np.array([df_features_h.loc['2015-01-06 00:00':'2016-12-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2015-01-06 00:00':'2016-12-31 23:00'].shape[0])])
            inputs['vali']['H_' + str(hour)]['2'] = np.array([df_features_h.loc['2017-01-01 00:00':'2017-02-28 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2017-01-01 00:00':'2017-02-28 23:00'].shape[0])])
            targets['train']['H_' + str(hour)]['2'] = df_targets_h.loc['2015-01-06 00:00':'2016-12-31 23:00']
            targets['vali']['H_' + str(hour)]['2'] = df_targets_h.loc['2017-01-01 00:00':'2017-02-28 23:00']

            inputs['train']['H_' + str(hour)]['3'] = np.array([df_features_h.loc['2015-01-06 00:00':'2017-02-28 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2015-01-06 00:00':'2017-02-28 23:00'].shape[0])])
            inputs['vali']['H_' + str(hour)]['3'] = np.array([df_features_h.loc['2017-03-01 00:00':'2017-04-30 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2017-03-01 00:00':'2017-04-30 23:00'].shape[0])])
            targets['train']['H_' + str(hour)]['3'] = df_targets_h.loc['2015-01-06 00:00':'2017-02-28 23:00']
            targets['vali']['H_' + str(hour)]['3'] = df_targets_h.loc['2017-03-01 00:00':'2017-04-30 23:00']

            inputs['train']['H_' + str(hour)]['4'] = np.array([df_features_h.loc['2015-01-06 00:00':'2017-04-30 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2015-01-06 00:00':'2017-04-30 23:00'].shape[0])])
            inputs['vali']['H_' + str(hour)]['4'] = np.array([df_features_h.loc['2017-05-01 00:00':'2017-06-30 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2017-05-01 00:00':'2017-06-30 23:00'].shape[0])])
            targets['train']['H_' + str(hour)]['4'] = df_targets_h.loc['2015-01-06 00:00':'2017-04-30 23:00']
            targets['vali']['H_' + str(hour)]['4'] = df_targets_h.loc['2017-05-01 00:00':'2017-06-30 23:00']

            inputs['train']['H_' + str(hour)]['5'] = np.array([df_features_h.loc['2015-01-06 00:00':'2017-06-30 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2015-01-06 00:00':'2017-06-30 23:00'].shape[0])])
            inputs['vali']['H_' + str(hour)]['5'] = np.array([df_features_h.loc['2017-07-01 00:00':'2017-08-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2017-07-01 00:00':'2017-08-31 23:00'].shape[0])])
            targets['train']['H_' + str(hour)]['5'] = df_targets_h.loc['2015-01-06 00:00':'2017-06-30 23:00']
            targets['vali']['H_' + str(hour)]['5'] = df_targets_h.loc['2017-07-01 00:00':'2017-08-31 23:00']

            inputs['train']['H_' + str(hour)]['6'] = np.array([df_features_h.loc['2015-01-06 00:00':'2017-08-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2015-01-06 00:00':'2017-08-31 23:00'].shape[0])])
            inputs['vali']['H_' + str(hour)]['6'] = np.array([df_features_h.loc['2017-09-01 00:00':'2017-10-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2017-09-01 00:00':'2017-10-31 23:00'].shape[0])])
            targets['train']['H_' + str(hour)]['6'] = df_targets_h.loc['2015-01-06 00:00':'2017-08-31 23:00']
            targets['vali']['H_' + str(hour)]['6'] = df_targets_h.loc['2017-09-01 00:00':'2017-10-31 23:00']

            # inputs['train']['H_'+str(hour)]['7'] = df_features_h.loc['2015-01-06 00:00':'2017-10-31 23:00']
            inputs['vali']['H_' + str(hour)]['7'] = np.array([df_features_h.loc['2017-11-01 00:00':'2017-12-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2017-11-01 00:00':'2017-12-31 23:00'].shape[0])])
            # targets['train']['H_'+str(hour)]['7'] = df_targets_h.loc['2015-01-06 00:00':'2017-10-31 23:00']
            targets['vali']['H_' + str(hour)]['7'] = df_targets_h.loc['2017-11-01 00:00':'2017-12-31 23:00']

            # inputs['train']['H_'+str(hour)]['8'] = df_features_h.loc['2015-01-06 00:00':'2017-12-31 23:00']
            inputs['vali']['H_' + str(hour)]['8'] = np.array([df_features_h.loc['2018-01-01 00:00':'2018-02-28 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2018-01-01 00:00':'2018-02-28 23:00'].shape[0])])
            # targets['train']['H_'+str(hour)]['8'] = df_targets_h.loc['2015-01-06 00:00':'2017-12-31 23:00']
            targets['vali']['H_' + str(hour)]['8'] = df_targets_h.loc['2018-01-01 00:00':'2018-02-28 23:00']

            # inputs['train']['H_'+str(hour)]['9'] = df_features_h.loc['2015-01-06 00:00':'2018-02-28 23:00']
            inputs['vali']['H_' + str(hour)]['9'] = np.array([df_features_h.loc['2018-03-01 00:00':'2018-04-30 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2018-03-01 00:00':'2018-04-30 23:00'].shape[0])])
            # targets['train']['H_'+str(hour)]['9'] = df_targets_h.loc['2015-01-06 00:00':'2018-02-28 23:00']
            targets['vali']['H_' + str(hour)]['9'] = df_targets_h.loc['2018-03-01 00:00':'2018-04-30 23:00']

            # inputs['train']['H_'+str(hour)]['10'] = df_features_h.loc['2015-01-06 00:00':'2018-04-30 23:00']
            inputs['vali']['H_' + str(hour)]['10'] = np.array([df_features_h.loc['2018-05-01 00:00':'2018-06-30 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2018-05-01 00:00':'2018-06-30 23:00'].shape[0])])
            # targets['train']['H_'+str(hour)]['10'] = df_targets_h.loc['2015-01-06 00:00':'2018-04-30 23:00']
            targets['vali']['H_' + str(hour)]['10'] = df_targets_h.loc['2018-05-01 00:00':'2018-06-30 23:00']

            # inputs['train']['H_'+str(hour)]['11'] = df_features_h.loc['2015-01-06 00:00':'2018-06-30 23:00']
            inputs['vali']['H_' + str(hour)]['11'] = np.array([df_features_h.loc['2018-07-01 00:00':'2018-08-31 23:00'].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2018-07-01 00:00':'2018-08-31 23:00'].shape[0])])
            # targets['train']['H_'+str(hour)]['11'] = df_targets_h.loc['2015-01-06 00:00':'2018-06-30 23:00']
            targets['vali']['H_' + str(hour)]['11'] = df_targets_h.loc['2018-07-01 00:00':'2018-08-31 23:00']

            # inputs['train']['H_'+str(hour)]['12'] = df_features_h.loc['2015-01-06 00:00':'2018-08-31 23:00']
            inputs['vali']['H_' + str(hour)]['12'] = np.array([df_features_h.loc['2018-09-01 00:00':].iloc[i].values.reshape(24, -1) for i in range(df_features_h.loc['2018-09-01 00:00':].shape[0])])
            # targets['train']['H_'+str(hour)]['12'] = df_targets_h.loc['2015-01-06 00:00':'2018-08-31 23:00']
            targets['vali']['H_' + str(hour)]['12'] = df_targets_h.loc['2018-09-01 00:00':]



    filename = os.path.join(DIR_DATA, 'inputs.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(inputs,f)

    filename = os.path.join(DIR_DATA, 'targets.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(targets, f)

    print('Done. Dataset saved in folder ./data')

    # for key1 in inputs.keys():
    #
    #     for key2 in inputs[key1].keys():
    #
    #         for hour in range(24):
    #
    #             if hour < 10:
    #                 hour_str = '0' + str(hour) + ':00'
    #             else:
    #                 hour_str = str(hour) + ':00'
    #
    #             print(key1 + '_' + key2 + ' ' + hour_str)
    #
    #             temp_input_df = inputs[key1][key2]
    #             temp_target_df = targets[key1][key2]
    #
    #             temp_hour_input_matrix = temp_input_df.loc[temp_input_df.index.strftime('%H:%M') == hour_str].values
    #             temp_hour_target_matrix = temp_target_df.loc[temp_target_df.index.strftime('%H:%M') == hour_str].values
    #
    #             temp_hour_index = temp_target_df.index
    #
    #             filename = str(hour) + '-' + key1 + '_' + key2
    #
    #             np.save(filename + '-inputs.npy', temp_hour_input_matrix)
    #             np.save(filename + '-targets.npy', temp_hour_target_matrix)
    #
    #             with open(filename + '-dates.pkl', 'wb') as f:
    #                 pickle.dump(temp_hour_index, f)

def load_data():

    if ('inputs.pkl' not in os.listdir(DIR_DATA)) or ('targets.pkl' not in os.listdir(DIR_DATA)):

        print('Generating data from %s file...' %(RAW_XLSX_FILENAME))
        generate_data(SCALER_TYPE)

    print('Loading data from ./data ...')
    filename = os.path.join(DIR_DATA, 'inputs.pkl')
    with open(filename, 'rb') as f:
        inputs = pickle.load(f)

    filename = os.path.join(DIR_DATA, 'targets.pkl')
    with open(filename, 'rb') as f:
        targets = pickle.load(f)
    print('Done.')
    return (inputs, targets)




