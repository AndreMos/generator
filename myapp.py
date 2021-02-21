import streamlit as st
#from transformers import AutoTokenizer, AutoModel
#import numpy as np
#import torch
#import torch.nn as nn
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
#import dlib
import pretrained_networks


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
    st.sidebar.button('Generate profile')

    # if submit:
    #     class_res = process(story)
    #     st.subheader(dicti[str(class_res)])
    #     #write

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
