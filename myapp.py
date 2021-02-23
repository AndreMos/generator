import streamlit as st
#from transformers import AutoTokenizer, AutoModel
#import numpy as np
#import torch
#import torch.nn as nn
import argparse
import numpy as np
import PIL.Image
# import dnnlib
# import dnnlib.tflib as tflib
import re
import sys
import dlib
# import pretrained_networks
import gdown

import os
import sys
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils import imwrite, immerge
from training.misc import load_pkl
import dnnlib
import dnnlib.tflib as tflib


def main():
    # story ='Я люблю тебя'
    # bert, classif,tokenizer = load_components()
    # pred_cl=pred(bert, classif,story,tokenizer)
    dicti={'0':'Досуг', '1':'Искусство и культура', '2':'Карьера','3': 'Коммуникации',
      '4': 'Наука','5': 'Обучение', '6':'Спорт', '7':'Стартапы'}
    col1, col2 = st.beta_columns(2)
    col1.title('Avatar here')
    #story = col2.text_area('Insert news')
    submit = col2.title('Meta-info here')#
    st.sidebar.text_input('Name', 'John')
    st.sidebar.text_input('Surname', 'Johnson')
    add_selectbox = st.sidebar.select_slider('Age', ['Child', 'Teen', 'Adult', 'Old'])
    st.sidebar.radio('Gender',['Male', "Female"])
    but = st.sidebar.button('Generate profile')
    if but:
        model = load()
        col1.title('image')

    # if submit:
    #     class_res = process(story)
    #     st.subheader(dicti[str(class_res)])
    #     #write
@st.cache(allow_output_mutation=True)
def load():
    print('Start')
    url = 'https://drive.google.com/uc?id=1H_H8GtJUbdM2PFapDZpnDRw7KPmU_lPh'
    output = 'modulus'
    gdown.download(url, output, quiet=False)
    print('End')
    tf_config = {'rnd.np_random_seed': 1000}
    tflib.init_tf(tf_config)
    _, _, _, Gs, _ = load_pkl('modulus')
    return Gs

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
