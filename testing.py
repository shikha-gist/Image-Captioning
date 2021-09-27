import random
from data import  TextField, RawField
from caption_model.transformer import Transformer
import torch 
import os, pickle,sys
import numpy as np
import itertools
from torch import nn
import multiprocessing
from shutil import copyfile
import warnings


from load_model_3layers import load_capModel
import re 
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

from detection import proposals,detection_detr 
import sys 

def main():

    detectionModel=detection_detr.load_detr() # 


    class Args():
        def __init__(self):
            self.device=torch.device('cuda:0')
            self.exp_name='LATGeO_transformer'
            self.batch_size=10
            self.workers=1
            self.m=40
            self.head=8
            self.label_file='./data/objects_vocab.txt'
            self.load_weight='./trained_models/rcnn_31_epoch.pth'
            self.detection='rcnn'
    args=Args()


    device = torch.device('cuda:0')
## 

    print('LATGeO Image Captioning Testing')
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    model=load_capModel(args,text_field,device)
    model.eval()

    def predict_caption(file_url,beam_size=1,show_objs=False):
        load_online=False
        if file_url.startswith('http'):
            im = Image.open(requests.get(file_url, stream=True).raw)
            im.save('./temp.jpg')
            file_url='./temp.jpg'
            load_online=True
        objs,bbox,probs,labels,im=proposals.process_image(file_url,detectionModel,text_field,args,50,device,show_objs)
        out, _ =model.beam_search(objs.to(device),bbox.to(device)
                        ,labels.to(device),probs.to(device),
                        20, text_field.vocab.stoi['<eos>'],beam_size, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        print("Caption of the image is:" , " ".join(caps_gen[0]))
        if show_objs==False:
            plt.imshow(im)
            plt.show()
        if load_online:
            os.remove(file_url)
        return 
    imgUrl='./sample_images/COCO_test2014_000000000245.jpg' #path to your image
    #objs,bbox,probs,labels,im=proposals.process_image(imgUrl,detectionModel,text_field,args,50,device,True)  #if you would like to show the proposals as well
    predict_caption(imgUrl,5,False)   # when the beam size is 5 and showing the proposal along with the caption
    



if __name__ == "__main__":
    main()

    
