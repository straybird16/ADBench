import pandas as pd
import numpy as np
import os


def get_columm_average(file_path:os.PathLike, num_tests_per_dataset:int=-1, usecols=1):
    experiment_results = pd.read_csv(file_path)  # read from file as pandas dataframe
    # get roc score array
    if num_tests_per_dataset==-1:
        num_tests_per_dataset = len(experiment_results.iloc[:,0])
    if type(usecols) == int:
        usecols = [i for i in range(usecols, len(experiment_results.iloc[0]))]
    # get column names in file
    column_names = list(experiment_results.columns)
    column_names[0]='Dataset name'
    # extract results (ROC score)
    result_array = np.array(experiment_results.iloc[:,usecols]).reshape(-1, num_tests_per_dataset, experiment_results.iloc[:,usecols].shape[1])
    result_array = np.nanmean(result_array, axis=1)
    # extract dataset names, one per repetition of experiments
    dataset_name_array = np.array(experiment_results.iloc[:,0])
    dataset_name_array = dataset_name_array[::num_tests_per_dataset]
    first_delimiter_index = np.array(list((map(lambda x :str.find(x, ','), dataset_name_array)))) - 1
    # 
    dataset_name_array = np.array([dataset_name_array[i][2:index] for i, index in enumerate(first_delimiter_index)]).reshape(-1, 1)
    # 
    results_df = pd.DataFrame(np.concatenate((dataset_name_array, result_array), axis=-1), columns=column_names)
    dataset_names = column_names[1:]
    dataset_names_roc_mean = [name+'_roc_mean' for name in dataset_names]
    dataset_names_roc_var = [name+'_roc_var' for name in dataset_names]
    mean_var_df = pd.DataFrame(np.concatenate((np.nanmean(result_array, axis=0, keepdims=True), np.nanvar(result_array, axis=0, keepdims=True)), axis=-1), columns=dataset_names_roc_mean+dataset_names_roc_var)
    #mean_var_df = pd.DataFrame({'ROC_mean':result_array.mean(), 'ROC_var': result_array.var()}, index=[0])
    new_df=pd.concat([results_df,mean_var_df], axis=1)
    return new_df
