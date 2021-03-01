import re
import sys
import os
import argparse
import zipfile
import logging
import base64
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import gdown

from train.save_bn import read_params, read_structure
from train.sampling import generate_synthetics, get_probability, sample
from libpgm.hybayesiannetwork import HyBayesianNetwork

def fold(age):
    age = int(age)
    if age <=20:
        return 'teen'
    elif age <= 45:
        return 'adult'
    else:
        return 'old'

def sam(bn):
    bn = bn.copy()
    counter = 3
    logging.info('BN shape: ',bn.shape)
    arr = []
    if bn[bn['has_high_education'] == '1'].shape[0] != 0:
        h1 = bn[bn['has_high_education'] == '1'].sample(1)
        bn.drop(h1.index.values, axis = 0, inplace = True)
        arr.append(h1)
        counter -= 1
    if bn[bn['relation'] != 0].shape[0] != '0':
        h2 = bn[bn['relation'] != '0'].sample(1)
        bn.drop(h2.index.values, axis = 0, inplace = True)
        arr.append(h2)
        counter -= 1
    h3 = bn.sample(counter)
    arr.append(h3)
    return pd.concat(arr, axis = 0)

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_data.csv">Download generated synthetic dataset</a>'
    return href

def reg_columns():
    return st.beta_columns(3)

def main():
    dicti_fam = {
    '0' : 'not specified',
    '1' : 'single',
    '2' : 'in a relationship',
    '3' : 'engaged',
    '4' : 'married',
    '5' : "it's complicated",
    '6' : 'actively searching',
    '7' : 'in love'
    }
    dicti_educ = {'0' : 'No', '1' : 'Yes'}
    dicti_gen = {'1' : 'female', '2' : 'male'}
    dataset, names, w_names, final_bn, idx_to_interest = load()

    slider = st.sidebar.selectbox('Age', ['not specified', 'Teen', 'Adult', 'Old'])
    li = ['teen', 'adult', 'old']
    gen = st.sidebar.radio('Gender', ['not specified', 'Male', "Female"])
    but = st.sidebar.button('Generate profiles')

    if but:
        age = slider.lower()
        if slider == 'not specified' and  gen == 'not specified':
            bn = sample(final_bn, None, None, names, white_names = w_names)
        elif age == 'not specified':
            bn = sample(final_bn, gender = gen, names = names, white_names = w_names)
        elif gen == 'not specified':
            bn = sample(final_bn, age = age, names = names, white_names = w_names)
        else:
            bn = sample(final_bn, age = age, gender = gen, names = names, white_names = w_names )
        res = sam(bn)
        logging.info("Synt shape: " + str(res.shape))


        for rec in np.array_split(res,res.shape[0]):
            col1, col2, col3 = reg_columns()
            name = rec['names'].iloc[0]
            educ = dicti_educ[rec['has_high_education'].iloc[0]]
            fam = dicti_fam[rec['relation'].iloc[0]]
            inter = rec.iloc[0,5:-1].sort_values( ascending = False)[:4].index.values
            gender = dicti_gen[rec['sex'].iloc[0]]
            age1 = rec['age'].iloc[0]
            col2.markdown(f'**Name** : {name}')
            col2.markdown(f'**Age: ** {age1}')
            col2.markdown(f'**Gender: ** {gender}')
            col2.markdown(f'**High education:** {educ}')
            col2.markdown(f'**Family status: ** {fam}')
            # col2.text(' ')
            # col2.text(' ')
            # col2.text(' ')
            # col2.text(' ')

            col3.markdown(f"**User interests:** ")
            for k in idx_to_interest[idx_to_interest['topic'].isin(inter)]['key_words'].values:
                col3.markdown(k)
            # col3.text(' ')
            # col3.text(' ')

            h = names[names['name'].str.find(name)!=-1]['type']
            if h.shape[0] > 0:
                race = h.iloc[0]
            else:
                race = 'White'
            try:
                logging.info((race, fold(age1),gender))
                ovr = dataset[(dataset['race'] == race) & (dataset['age'] == fold(age1)) & (dataset['gender'] == gender)]
                logging.info(ovr.shape)#
                ovr = ovr.sample(1)
            except:
                if fold(age1) == 'teen':
                    ovr = dataset[(dataset['race'] == race) & (dataset['age'] == 'child') & (dataset['gender'] == gender)].sample(1)
            res = ovr['id'].iloc[0]
            path_img = Path(__file__).parent / 'new_generator1'
            avatar = Image.open(path_img / race  /  (str(res) + '.jpg'))
            col1.image(avatar, caption = 'Profile picture')
            col1.text(' ')
        st.markdown(get_table_download_link(bn), unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load():
    path = Path(__file__).parent
    skel = read_structure('K2_bn_structure')
    params = read_params('K2_bn_param')
    final_bn = HyBayesianNetwork(skel, params)
    logging.info('BN downloaded')

    idx_to_interest = pd.read_csv('key_words_groups_interests.csv')
    white_names = pd.read_csv(path / 'data/white_names.csv')


    url = 'https://drive.google.com/uc?id=1NQIXdma7J0HJ6WfU0Z7VUcF_ypoJ9UrN'
    output = 'modulus.zip'
    logging.info('Start load')
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile('modulus.zip', 'r') as zip_ref:
        zip_ref.extractall()

    path_f = Path(__file__).parent / 'new_generator1'
    dataset = pd.read_csv(path_f / 'dataset.csv')
    dataset['id'] = dataset['id'].astype(int)
    names = pd.read_csv(path_f / 'names.csv')

    return dataset, names, white_names, final_bn, idx_to_interest





if __name__ == "__main__":
    main()
