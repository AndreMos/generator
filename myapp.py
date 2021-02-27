import streamlit as st
import argparse
import numpy as np
from PIL import Image
import re
import sys
import pandas as pd
import gdown

import os
import sys
import argparse

import zipfile
import logging
os.chdir('/app/generator/')
#os.chdir('/app/generator/')
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
    # else:
    # return 3

def main():
    # story ='Я люблю тебя'
    # bert, classif,tokenizer = load_components()
    # pred_cl=pred(bert, classif,story,tokenizer)
    # dicti={'0':'Досуг', '1':'Искусство и культура', '2':'Карьера','3': 'Коммуникации',
    #   '4': 'Наука','5': 'Обучение', '6':'Спорт', '7':'Стартапы'}
    os.chdir('/app/generator/')
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
    dataset, names, final_bn, idx_to_interest = load()

    #df = sample(final_bn)
    #df
    col1, col2, col3 = st.beta_columns(3)
    #col1.title('Avatar here')
    #story = col2.text_area('Insert news')
    #col2.title('Meta-info here')#
    #name = st.sidebar.text_input('Name', 'John')

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
            bn = sample(final_bn)
        elif age == 'not specified':
            bn = sample(final_bn, gender = gen)
        elif gen == 'not specified':
            bn = sample(final_bn, age = age)
        else:
            bn = sample(final_bn, age = age, gender = gen)
        res = bn.sample(3)
        res1 = res
        #bn.T.iloc[:,4:-1]
        for rec in np.array_split(res,3):
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

            col3.markdown(f"**User interests:** {' '.join(idx_to_interest[idx_to_interest['topic'].isin(inter)]['key_words'].values)}")
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
                #logging.info((race, fold(age1),gender))
                #(race, fold(age1),gender)
                if fold(age1) == 'teen':
                ovr = dataset[(dataset['race'] == race) & (dataset['age'] == 'child') & (dataset['gender'] == gender)].sample(1)
            res = ovr['id'].iloc[0]
            avatar = Image.open(race + '/' + str(res) + '.jpg')
            col1.image(avatar, caption = 'Profile picture')
            col1.text(' ')
            os.chdir('/app/generator/')
            res1
        # #age = slider.lower()
        # gender = gen.lower()
        # #logging.info([gender,age,race])
        # try:
        #     ovr = dataset[(dataset['race'] == race) & (dataset['age'] == age) & (dataset['gender'] == gender)].sample(4)
        # except:
        #     new_age_idx = li.index(age)
        #     if new_age_idx == len(li) - 1:
        #         new_age = li[new_age_idx - 1]
        #         ovr = dataset[(dataset['race'] == race) & (dataset['age'] == new_age) & (dataset['gender'] == gender)].sample(4)
        #     else:
        #         new_age = li[new_age_idx + 1]
        #         ovr = dataset[(dataset['race'] == race) & (dataset['age'] == new_age) & (dataset['gender'] == gender)].sample(4)
        # res = ovr['id'].iloc[0]
        #
        # avatar = Image.open(race + '/' + str(res) + '.jpg')
        # pl = col1.empty()
        # pl.image(avatar, caption = 'Profile picture')
        # col2.subheader("Name: " + name)
        # col2.subheader('Surname: ' + sn)
        # col2.subheader('Location: ' )
        # col3.subheader('Other meta-data')
        # show_alt1 = show_alt.button('Show alternative')
        #     #alt = st.sidebar.selectbox('Alternative profile ', ['current', '1','2','3'])
        #     #
        # if show_alt1:
        #     res = ovr['id'].iloc[1]
        #     logging.info(res)
        #     avatar = Image.open(race + '/' + str(res) + '.jpg')
        #     pl.image(avatar, caption = 'Profile picture')

        # if alt == '1':
        #     res = ovr['id'].iloc[1]
        #     avatar = Image.open(race + '/' + str(res) + '.jpg')
        #     pl.image(avatar, caption = 'Profile picture')
        # elif alt == '2':
        #     res = ovr['id'].iloc[2]
        #     avatar = Image.open(race + '/' + str(res) + '.jpg')
        #     pl.image(avatar, caption = 'Profile picture')
        # elif alt == '3':
        #     res = ovr['id'].iloc[3]
        #     avatar = Image.open(race + '/' + str(res) + '.jpg')
        #     pl.image(avatar, caption = 'Profile picture')




    # if submit:
    #     class_res = process(story)
    #     st.subheader(dicti[str(class_res)])
    #     #write
@st.cache(allow_output_mutation=True)
def load():
    skel = read_structure('K2_bn_structure')
    params = read_params('K2_bn_param')
    final_bn = HyBayesianNetwork(skel, params)
    idx_to_interest = pd.read_csv('key_words_groups_interests.csv')
    logging.info('BN downloaded')
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

    return dataset, names, final_bn, idx_to_interest

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
