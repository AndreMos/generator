import streamlit as st
#from transformers import AutoTokenizer, AutoModel
#import numpy as np
#import torch
#import torch.nn as nn
import argparse
import numpy as np
from PIL import Image
# import dnnlib
# import dnnlib.tflib as tflib
import re
import sys
import pandas as pd
# import pretrained_networks
import gdown

import os
import sys
import argparse

import zipfile
import logging
#os.chdir('/app/generator/')




def main():
    # story ='Я люблю тебя'
    # bert, classif,tokenizer = load_components()
    # pred_cl=pred(bert, classif,story,tokenizer)
    # dicti={'0':'Досуг', '1':'Искусство и культура', '2':'Карьера','3': 'Коммуникации',
    #   '4': 'Наука','5': 'Обучение', '6':'Спорт', '7':'Стартапы'}
    dataset, names = load()

    col1, col2, col3 = st.beta_columns(3)
    #col1.title('Avatar here')
    #story = col2.text_area('Insert news')
    #col2.title('Meta-info here')#
    name = st.sidebar.text_input('Name', 'John')
    sn = st.sidebar.text_input('Surname', 'Johnson')
    slider = st.sidebar.select_slider('Age', ['Child', 'Teen', 'Adult', 'Old'])
    gen = st.sidebar.radio('Gender',['Male', "Female"])
    but = st.sidebar.button('Generate profile')

    os.chdir('/app/generator/new_generator1')
    if but:
        h = names[names['name'].str.find(name)!=-1]['type']
        if h.shape[0] > 0:
            race = h.iloc[0]
        else:
            race = 'White'
        age = slider.lower()
        gender = gen.lower()
        logging.info([gender,age,race])
        ovr = dataset[(dataset['race'] == race) & (dataset['age'] == age) & (dataset['gender'] == gender)].sample(4)
        res = ovr['id'].iloc[0]

        avatar = Image.open(race + '/' + str(res) + '.jpg')
        pl = col1.empty()
        pl.image(avatar, caption = 'Profile picture')
        col2.subheader("Name: " + name)
        col2.subheader('Surname: ' + sn)
        col2.subheader('Location: ' )
        col3.subheader('Other meta-data')

        alt = st.sidebar.selectbox('Alternative profile ', ['current', '1','2','3'])
        show_alt = st.sidebar.button('Show alternative')
        if show_alt:

            if alt == '1':
                res = ovr['id'].iloc[1]
                avatar = Image.open(race + '/' + str(res) + '.jpg')
                pl.image(avatar, caption = 'Profile picture')
            elif alt == '2':
                res = ovr['id'].iloc[2]
                avatar = Image.open(race + '/' + str(res) + '.jpg')
                pl.image(avatar, caption = 'Profile picture')
            elif alt == '3':
                res = ovr['id'].iloc[3]
                avatar = Image.open(race + '/' + str(res) + '.jpg')
                pl.image(avatar, caption = 'Profile picture')

    os.chdir('/app/generator/')


    # if submit:
    #     class_res = process(story)
    #     st.subheader(dicti[str(class_res)])
    #     #write
@st.cache(allow_output_mutation=True)
def load():
    url = 'https://drive.google.com/uc?id=1NQIXdma7J0HJ6WfU0Z7VUcF_ypoJ9UrN'
    output = 'modulus.zip'
    logging.info('Start load')
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile('modulus.zip', 'r') as zip_ref:
        zip_ref.extractall()

    stri = '/app/generator/new_generator1/'
    dataset = pd.read_csv(stri + 'dataset.csv')
    dataset['id'] = dataset['id'].astype(int)
    names = pd.read_csv(stri + 'names.csv')
    return dataset, names

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
