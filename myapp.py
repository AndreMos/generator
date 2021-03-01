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
    #os.chdir('/app/generator/')
    path = Path(__file__).parent
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

    #sn = st.sidebar.text_input('Surname', 'Johnson')
    slider = st.sidebar.selectbox('Age', ['not specified', 'Teen', 'Adult', 'Old'])
    #slider = st.sidebar.select_slider('Age', ['Child', 'Teen', 'Adult', 'Old'])
    li = ['teen', 'adult', 'old']
    gen = st.sidebar.radio('Gender', ['not specified', 'Male', "Female"])
    but = st.sidebar.button('Generate profiles')

    if but:
        age = slider.lower()
        #gender = gen.lower()
        #sample

        if slider == 'not specified' and  gen == 'not specified':
            bn = sample(final_bn, None, None, names, white_names = w_names)
        elif age == 'not specified':
            bn = sample(final_bn, gender = gen, names = names, white_names = w_names)
        elif gen == 'not specified':
            bn = sample(final_bn, age = age, names = names, white_names = w_names)
        else:
            bn = sample(final_bn, age = age, gender = gen, names = names, white_names = w_names )
        res = sam(bn)
        res1 = res
        logging.info("Synt shape: " + str(res.shape))


        for rec in np.array_split(res,res.shape[0]):
            col1, col2, col3 = reg_columns()
            name = rec['names'].iloc[0]
            educ = dicti_educ[rec['has_high_education'].iloc[0]]
            fam = dicti_fam[rec['relation'].iloc[0]]
            inter = rec.iloc[0,5:-1].sort_values( ascending = False)[:4].index.values
            gender = dicti_gen[rec['sex'].iloc[0]]

            #inter
            #inter.sort_values( ascending = False)[:4]
            age1 = rec['age'].iloc[0]
            #aage
            col2.markdown(f'**Name** : {name}')
            col2.markdown(f'**Age: ** {age1}')
            col2.markdown(f'**Gender: ** {gender}')
            col2.markdown(f'**High education:** {educ}')
            col2.markdown(f'**Family status: ** {fam}')
            col2.text(' ')
            col2.text(' ')
            col2.text(' ')
            col2.text(' ')

            col3.markdown(f"**User interests:** ")
            for k in idx_to_interest[idx_to_interest['topic'].isin(inter)]['key_words'].values:
                col3.markdown(k)
            col3.text(' ')
            col3.text(' ')

            os.chdir('/app/generator/new_generator1')
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
                os.chdir('/app/generator/')
                if fold(age1) == 'teen':
                    ovr = dataset[(dataset['race'] == race) & (dataset['age'] == 'child') & (dataset['gender'] == gender)].sample(1)
            os.chdir('/app/generator/new_generator1')
            res = ovr['id'].iloc[0]
            avatar = Image.open(race + '/' + str(res) + '.jpg')
            col1.image(avatar, caption = 'Profile picture')
            col1.text(' ')
            os.chdir('/app/generator/')
        st.markdown(get_table_download_link(bn), unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load():
    path = Path(__file__).parent
    skel = read_structure('K2_bn_structure')
    params = read_params('K2_bn_param')
    final_bn = HyBayesianNetwork(skel, params)
    idx_to_interest = pd.read_csv('key_words_groups_interests.csv')

    white_names = pd.read_csv(path / 'data/white_names.csv')
    logging.info('BN downloaded')
    url = 'https://drive.google.com/uc?id=1NQIXdma7J0HJ6WfU0Z7VUcF_ypoJ9UrN'
    output = 'modulus.zip'
    logging.info('Start load')
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile('modulus.zip', 'r') as zip_ref:
        zip_ref.extractall()

    path_f = Path(__file__).parent / 'new_generator1'
    #stri = '/app/generator/new_generator1/'
    dataset = pd.read_csv(path_f + 'dataset.csv')
    dataset['id'] = dataset['id'].astype(int)
    names = pd.read_csv(stri + 'names.csv')

    return dataset, names, white_names, final_bn, idx_to_interest

    #return Gs

# def generate_images(network_pkl, seeds, truncation_psi):
#     print('Loading networks from "%s"...' % 'gdrive:networks/stylegan2-ffhq-config-f.pkl')
#     _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-ffhq-config-f.pkl')
#     noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
#
#     Gs_kwargs = dnnlib.EasyDict()
#     Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
#     Gs_kwargs.randomize_noise = False
#     if truncation_psi is not None:
#         Gs_kwargs.truncation_psi = truncation_psi
#     os.chdir(r'/content/gdrive/My Drive/maga/new_data')
#     for seed_idx, seed in enumerate(seeds):
#         print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
#         rnd = np.random.RandomState(seed)
#         z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
#         tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
#         images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
#         return PIL.Image.fromarray(images[0], 'RGB')
        #return images
        #PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))


# def process(story):
#     bert, classif,tokenizer = load_components()
#     pred_cl=pred(bert, classif,story,tokenizer)
#     return pred_cl.item()
#
#
# class Net(nn.Module):
#     def __init__(self, num_feature):
#         super(Net, self).__init__()
#         self.layer_1 = nn.Linear(num_feature, 128)
#         self.layer_2 = nn.Linear(128, 8)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.layer_1(x)
#         x = self.relu(x)
#         x = self.layer_2(x)
#         return x
#
# @st.cache(allow_output_mutation=True)
# def load_components():
#     tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
#     model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
#     model_new=Net(768)
#     model_new.load_state_dict(torch.load('model',map_location=torch.device('cpu')) )
#     model_new.eval()
#     return model,model_new,tokenizer
#
#
#
# def pred(model,model_new,x,tokenizer):
#     pt_batch = tokenizer(
#     x,
#     padding=True,
#     truncation=True,
#     return_tensors="pt")
#
#     pt_outputs = model(**pt_batch)
#     out=model_new(pt_outputs[1])
#     cc=torch.log_softmax(out,dim = 1)
#     _, y_pred_tags = torch.max(cc, dim = 1)
#     return y_pred_tags

if __name__ == "__main__":
    main()
