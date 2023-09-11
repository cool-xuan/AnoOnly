# %% [markdown]
# # Run ADBench 
# - Here we provide a demo for testing AD algorithms on the datasets proposed in ADBench.
# - Feel free to evaluate any customized algorithm in ADBench.
# - For reproducing the complete experiment results in ADBench, please run the code in the run.py file.

# %%
# import basic package
import os
import datetime
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# import the necessary package
from data_generator import DataGenerator
from myutils import Utils

datagenerator = DataGenerator() # data generator
utils = Utils() # utils function


from baseline.FEAWAD.run import FEAWAD
from baseline.PReNet.run import PReNet
from baseline.DevNet.run import DevNet
from baseline.DeepSAD.src.run import DeepSAD

import argparse 

# image datasets
image_data_files = os.listdir('datasets/CV_by_ResNet18')
cifar10_list = [data_file[:-4] for data_file in image_data_files if 'CIFAR10' in data_file]

image_datasets_dict = {
    'Image_cifar10': cifar10_list,
    }
text_datasets_dict = {
    'Text_amazon': ['amazon', ],
    'Text_imdb': ['imdb', ],
    'Text_yelp': ['yelp', ],
    }

model_dict = {
    'FEAWAD': FEAWAD,
    'PReNet': PReNet,
    'DevNet': DevNet,
    'DeepSAD': DeepSAD,
    }


def calculate_auc(mdoel_name, datasets_dict, data_types, df_aucroc_dict, df_ret, avg_weight=False, column=None):
    column = column + '_weight' if avg_weight else column
    avg_auc = {k: 0.0 for k in data_types}
    idx = 0
    div_by = {k: 1e-6 for k in data_types}
    for datasets_name in datasets_dict:
        if datasets_name not in eval('{}_datasets_dict'.format(data_types[idx])):
            idx += 1
        curr_mean = df_aucroc_dict[datasets_name].mean().item()
        curr_len = len(datasets_dict[datasets_name]) if avg_weight else 1
        avg_auc[data_types[idx]] += curr_mean * curr_len
        div_by[data_types[idx]] += curr_len
        # save dataset-wise results 
        df_ret.loc[datasets_name, column] = curr_mean
    
    for data_type in data_types:
        df_ret.loc[data_type, column] = avg_auc[data_type] / div_by[data_type]
    
    df_ret.loc['total', column] = sum(avg_auc.values()) / sum(div_by.values())

def main(args):

    data_types = args.data_types
    datasets_dict = {}
    for data_type in data_types:
        datasets_dict.update(eval('{}_datasets_dict'.format(data_type)))

    # save the results
    eval_metrics = [
        'aucroc'      ,
        'aucpr'       ,
        'aucpr_normal',
    ]
    _model_dict = {args.baseline: model_dict[args.baseline]}
    dict_df_metrics = {}
    for metric in eval_metrics:
        dict_df_metrics[metric] = {}
        for name in datasets_dict:
            dict_df_metrics[metric][name] = pd.DataFrame(data=None, index=datasets_dict[name], columns = _model_dict.keys())

    # seed for reproducible results
    seed = 42

    for datasets_name in datasets_dict:
        for dataset in datasets_dict[datasets_name]:
            '''
            la: ratio of labeled anomalies, from 0.0 to 1.0
            realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
            noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
            '''
            
            # import the dataset
            datagenerator.dataset = dataset # specify the dataset name
            data = datagenerator.generator(la=args.la, realistic_synthetic_mode=None, noise_type=None, at_least_one_labeled=True, normal_clean=args.normal_clean, backbone=args.backbone) # only 10% labeled anomalies are available
            
            for name, clf in _model_dict.items():
                # model initialization
                print("Training {} (model) on {} (dataset) ...".format(name, dataset))
                clf = clf(
                    seed=seed,
                    model_name=name, 
                    anomaly_only=args.anomaly_only,
                )
                
                # training, for unsupervised models the y label will be discarded
                clf = clf.fit(X_train=data['X_train'], y_train=data['y_train']) # 
                
                # output predicted anomaly score on testing set
                score = clf.predict_score(data['X_test'])

                # evaluation
                result = utils.metric(y_true=data['y_test'], y_score=score)
                
                # save results
                for metric in eval_metrics:
                    if metric not in result: continue
                    dict_df_metrics[metric][datasets_name].loc[dataset, name] = result[metric]
                

    index = list(datasets_dict.keys())
    index += data_types
    index += ['total']
    
    df_ret = pd.DataFrame(index=index, 
                          columns=eval_metrics)
    for metric in eval_metrics:
        calculate_auc(args.baseline, datasets_dict, data_types, dict_df_metrics[metric], avg_weight=False, df_ret=df_ret, column=metric)
    
    df_ret = df_ret.applymap(lambda x: '{:.2%}'.format(x))
    if 'threshold' in df_ret.columns:
        df_ret[['threshold']] = df_ret[['threshold']].applymap(lambda x: float(x.strip("%"))/100)
    print("\n"+df_ret.to_markdown())

def get_parser():

    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--baseline', type=str, default='PReNet') # FEAWAD, PReNet, DevNet, DeepSAD
    # dataset
    parser.add_argument('--data_types', type=str, nargs='+', default=['image', 'text'])
    parser.add_argument('--backbone', type=str, default='small', choices=['small', 'large']) 
    parser.add_argument('--la', type=float, default=0.1)
    parser.add_argument('--normal_clean', action='store_true')
    # method
    parser.add_argument('--anomaly_only', action='store_true')

    args = parser.parse_args()
    
    return args

if __name__ =='__main__':
    args = get_parser()
    print(args)
    main(args)
