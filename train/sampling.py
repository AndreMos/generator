from libpgm.hybayesiannetwork import HyBayesianNetwork
import numpy as np
import pandas as pd
import math
import random


def sample(bn: HyBayesianNetwork, age: str = None, gender: str = None, names: pd.DataFrame = None, white_names: pd.DataFrame = None ) -> pd.DataFrame:
    dataset = pd.DataFrame()
    age_values = []
    gender_value = 0
    # names = pd.read_csv('data/names.csv')
    # white_names = pd.read_csv('data/white_names.csv')
    white_names.iloc[:,1] = white_names.iloc[:,1].astype('int')
    if (age == None) & (gender == None):
        dataset = generate_synthetics(bn)
        if (dataset.shape[0] % 2 == 0):
            names = names['name'].tolist()
            names1 = random.sample(names, int(dataset.shape[0]/2))
            names2 = []
            for i in range(int(dataset.shape[0]/2),dataset.shape[0]):
                if dataset.loc[i,'sex'] == '1':
                    female_names = white_names.loc[white_names['sex'] == 1]
                    names2.append(random.choice(female_names['first_name'].tolist()))
                else:
                    male_names = white_names.loc[white_names['sex'] == 2]
                    names2.append(random.choice(male_names['first_name'].tolist()))
            names = names1+names2
            dataset['names'] = names
        else:
            names = names['name'].tolist()
            names1 = random.sample(names, dataset.shape[0]//2)
            names2 = []
            for i in range(dataset.shape[0]//2,dataset.shape[0]):
                if dataset.loc[i,'sex'] == '1':
                    female_names = white_names.loc[white_names['sex'] == 1]
                    names2.append(random.choice(female_names['first_name'].tolist()))
                else:
                    male_names = white_names.loc[white_names['sex'] == 2]
                    names2.append(random.choice(male_names['first_name'].tolist()))
            names = names1+names2
            dataset['names'] = names

    else:
        if age != None:
            if age == 'teen':
                age_values = [15,16,17,18,19,20]
            if age == 'adult':

                age_values = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
            if age == 'old':
                age_values = [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]
        if gender != None:
            if gender == 'Male':
                gender_value = 2
            else:
                gender_value = 1
        if (age != None) & (gender != None):
            df1 = generate_synthetics(bn, 200, evidence={'age': str(random.choice(age_values)), 'sex': str(gender_value)})
            df2 = generate_synthetics(bn, 200, evidence={'age': str(random.choice(age_values)), 'sex': str(gender_value)})
            df3 = generate_synthetics(bn, 200, evidence={'age': str(random.choice(age_values)), 'sex': str(gender_value)})
            dataset = pd.concat([df1, df2, df3])
            dataset.reset_index(inplace=True, drop=True)
        if (age != None) & (gender == None):
            df1 = generate_synthetics(bn, 200, evidence={'age': str(random.choice(age_values))})
            df2 = generate_synthetics(bn, 200, evidence={'age': str(random.choice(age_values))})
            df3 = generate_synthetics(bn, 200, evidence={'age': str(random.choice(age_values))})
            dataset = pd.concat([df1, df2, df3])
            dataset.reset_index(inplace=True, drop=True)
        if (age == None) & (gender != None):
            df1 = generate_synthetics(bn, 200, evidence={'sex': str(gender_value)})
            df2 = generate_synthetics(bn, 200, evidence={'sex': str(gender_value)})
            df3 = generate_synthetics(bn, 200, evidence={'sex': str(gender_value)})
            dataset = pd.concat([df1, df2, df3])
            dataset.reset_index(inplace=True, drop=True)

        if (dataset.shape[0] % 2 == 0):
            names = names['name'].tolist()
            names1 = random.sample(names, int(dataset.shape[0]/2))
            names2 = []
            for i in range(int(dataset.shape[0]/2),dataset.shape[0]):
                if dataset.loc[i,'sex'] == '1':
                    female_names = white_names.loc[white_names['sex'] == 1]
                    names2.append(random.choice(female_names['first_name'].tolist()))
                else:
                    male_names = white_names.loc[white_names['sex'] == 2]
                    names2.append(random.choice(male_names['first_name'].tolist()))
            names = names1+names2
            dataset['names'] = names
        else:
            names = names['name'].tolist()
            names1 = random.sample(names, dataset.shape[0]//2)
            names2 = []
            for i in range(dataset.shape[0]//2,dataset.shape[0]):
                if dataset.loc[i,'sex'] == '1':
                    female_names = white_names.loc[white_names['sex'] == 1]
                    names2.append(random.choice(female_names['first_name'].tolist()))
                else:
                    male_names = white_names.loc[white_names['sex'] == 2]
                    names2.append(random.choice(male_names['first_name'].tolist()))
            names = names1+names2
            dataset['names'] = names


    return dataset
























def generate_synthetics(bn: HyBayesianNetwork, n: int = 1000, evidence: dict = None) -> pd.DataFrame:
    """Function for sampling from BN

    Args:
        bn (HyBayesianNetwork): learnt BN
        n (int, optional): number of samples (rows). Defaults to 1000.
        evidence (dict): dictionary with values of params that initialize nodes

    Returns:
        pd.DataFrame: final sample
    """
    sample = pd.DataFrame()

    if evidence:
        sample = pd.DataFrame(bn.randomsample(10 * n, evidence=evidence))
        cont_nodes = []
        for key in bn.nodes.keys():
            if (str(type(bn.nodes[key])).split('.')[1] == 'lg') | (str(type(bn.nodes[key])).split('.')[1] == 'lgandd'):
                cont_nodes.append(key)
        sample.dropna(inplace=True)
        sample = sample.loc[(sample.loc[:, cont_nodes].values >= 0).all(axis=1)]
        sample.reset_index(inplace=True, drop=True)
    else:
        sample = pd.DataFrame(bn.randomsample(10 * n))
        cont_nodes = []
        for key in bn.nodes.keys():
            if (str(type(bn.nodes[key])).split('.')[1] == 'lg') | (str(type(bn.nodes[key])).split('.')[1] == 'lgandd'):
                cont_nodes.append(key)
        sample.dropna(inplace=True)
        sample = sample.loc[(sample.loc[:, cont_nodes].values >= 0).all(axis=1)]
        sample.reset_index(inplace=True, drop=True)




    # final_sample = pd.DataFrame()

    # i = 0
    # while i < n:
    #     sample = pd.DataFrame(bn.randomsample(1))
    #     flag = True
    #     for node in cont_nodes:
    #         if (sample.loc[0,node] < 0) | (str(sample.loc[0,node]) == 'nan'):
    #             flag = False
    #     if flag:
    #         final_sample = pd.concat([final_sample, sample])
    #         i = i + 1
    #     else:
    #         continue
    return sample

def get_probability(sample: pd.DataFrame, initial_data: pd.DataFrame, parameter: str) -> dict:
    """Helper function for calculation probability
       of each label in a sample. Also calculate
       confidence interval for a probability

    Args:
        sample (pd.DataFrame): Data sampled from a bayesian network
        initial_data (pd.DataFrame): Source encoded dataset
        parameter (str): Name of the parameter in which
        we want to calculate probabilities
        of labels

    Returns:
        dict: Dictionary in which
        key - is a label
        value - is a list [lower bound of the interval, probability, higher bound of the interval]
    """
    dict_prob = dict([(str(n), []) for n in initial_data[parameter].unique()])

    for i in dict_prob:
        grouped = sample.groupby(parameter)[parameter].count()
        grouped = {str(key): value for key, value in grouped.items()}
        if i in grouped:
            p = (grouped[i]) / sample.shape[0]
            std = 1.96 * math.sqrt(((1 - p) * p) / sample.shape[0])
            start = p - std
            end = p + std
            dict_prob[i].append(start)
            dict_prob[i].append(p)
            dict_prob[i].append(end)
        else:
            dict_prob[i].append(0)
            dict_prob[i].append(0)
            dict_prob[i].append(0)

    return dict_prob
