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
import detection.detection_rcnn as rcnn
from detection.proposals import plot_results_GUI
import glob
import io
import os
import PySimpleGUI as gui
from PIL import Image, ImageTk
import traceback
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

detectionModel=detection_detr.load_detr() 



class Args():
    def __init__(self):
        self.device=torch.device('cuda:0')
        self.exp_name='LATGeO_transformer'
        self.batch_size=10
        self.workers=1
        self.m=40
        self.head=8
        self.label_file='./data/objects_vocab.txt'
        self.load_weight='./trained_models/rcnn_31_epoch.pth'   #loading RCNN model
        self.detection='rcnn'
args=Args()



device = torch.device('cuda:0')
## 

print('LATGeO Image Captioning GUI Testing')
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
    if show_objs==False:
        plt.imshow(im)
    if load_online:
        os.remove(file_url)
    return " ".join(caps_gen[0])




def parse_folder(path):
    images = glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.png')
    return images

def load_image(path, window):  # for loading images after image path selection
    try:
        image = Image.open(path)
        image.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
    except:
        print(f"Unable to open {path}!")



        
def load_proposals_image(url,window):   ## for displaying proposals
    try:
        objs,bboxes,probs,labels,im=rcnn.get_detections(url)
   
        plot_im=plot_results_GUI(im,probs,bboxes,rcnn.classes)
        
        plot_im.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(plot_im)
        window["image_proposal"].update(data=photo_img)
    except:
        print(f"Unable to open!")



#  Two columns window layout 

def main():

    # For getting the file location
    file_list_column = [

        [

            gui.Text("Image Folder"),

            gui.In(size=(25, 1), enable_events=True, key="-FOLDER-"),

            gui.FolderBrowse(initial_folder='./sample_images'),


        ],

	# list all available figures in the selected directory
        [

            gui.Listbox(

                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-",select_mode = 'single', bind_return_key = True, change_submits= True

            ),
            gui.Button("Load Image",key="-LOAD-")

        ],

    ]


    # Window to show the image, image proposals and generated captions and selecting next and previous image.

    image_viewer_column = [

        [gui.Text("Choose an image :")],   

        [gui.Image(key="image"),gui.Image(key="image_proposal")],



        [gui.Text(size=(70, 1), key="-CAPTION-",justification='left')],
        [
                gui.Button("Object Proposals"),
                gui.Button("Generate Caption"),

        ],

        [
                gui.Button("Prev",key="-PREV-"),
                gui.Button("Next",key="-NEXT-")
        ],
        [
                gui.Button("EXIT",key="-EXIT-")

        ],

    ]


    # ----- Full LATGeO Image Captioning layout -----

    layout = [

        [

            gui.Column(file_list_column),

            gui.VSeperator(),

            gui.Column(image_viewer_column),

        ]

    ]
    # gui.theme('Dark Brown 1')

    window = gui.Window("LATGeO Image Captioning", layout)
    images = []
    location = 0

    # Run the Event Loop
    
    while True:

        event, values = window.read()

        if event == "Exit" or event=="-EXIT-" or event == gui.WIN_CLOSED:

            break

        # Folder name was filled in, make a list of files in the folder

        if event == "-FOLDER-":

            folder = values["-FOLDER-"]
           

            try:

                # Get list of files in folder

                file_list = os.listdir(folder)

            except:

                file_list = []


            fnames = [

                f

                for f in file_list

                if os.path.isfile(os.path.join(folder, f))

                and f.lower().endswith((".png", ".gif",".jpg",".jpeg"))

            ]

            window["-FILE LIST-"].update(fnames)
            
       

            
        if event == "-LOAD-":  # A file was chosen from the listbox and will load the image 

            try:

                filename = os.path.join(

                    values["-FOLDER-"], values["-FILE LIST-"][0]

                )
                
                
                load_image(filename, window)
                current_selection_index=window.Element("-FILE LIST-").GetIndexes()[0]
                images = parse_folder(values["-FOLDER-"])


            except Exception as e:

                print(f"Unable to open given path or No Image file is selected")
                 
                tb = traceback.format_exc()

                gui.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)
                    
                
        
        if event == "-NEXT-":  #choosing the next image from the list
            try:
                if location == len(images) - 1:
                    location = 0
                    current_selection_index = 0
                    
                else:
                    location += 1
                    current_selection_index = (current_selection_index + 1) 
                   
                load_image(images[location], window)
                filename= images[location]
                
                window.Element("-FILE LIST-").Update(set_to_index=current_selection_index)

            except Exception as e:

                print(f"Unable to open given path or No Next image present")
                 
                tb = traceback.format_exc()

                gui.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)
        if event == "-PREV-":    #choosing the previous image from the list
            try: 
                if location == 0:
                    location = len(images) - 1
                    current_selection_index = len(images) - 1
                else:
                    location -= 1
                    current_selection_index = (current_selection_index - 1) 
                filename= images[location]
                load_image(images[location], window)

                window.Element("-FILE LIST-").Update(set_to_index=current_selection_index)

            except Exception as e:

                print(f"Unable to open given path or No Previous image present")
                 
                tb = traceback.format_exc()

                gui.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)
    
        if event =="Generate Caption":   # for generating the caption of the image
            try:
                if filename:
                    url = filename
                    caps=predict_caption(url)
                    
                    caps=predict_caption(url,beam_size=5,show_objs=False)
                    window["-CAPTION-"].update(caps,background_color='#856ff8',font=("Times New Roman",12),text_color='Yellow')

                else :

                    window["-CAPTION-"].update("No image is selected.",background_color='#856ff8',font=("Times New Roman",11),text_color='Red')
                    gui.popup('Caption',"NO image is selected")

            except Exception as e:

                print(f"Unable to open given path or No Previous image present")

                tb = traceback.format_exc()

                gui.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)
            
        if event =="Object Proposals":   # for displaying the proposal of the image
            try:
                if filename:
                    url = filename
                    
                    
                    load_proposals_image(url,window)
                    
                else :
 
                    
                    gui.popup('Caption',"NO image is selected")
              
            except Exception as e:

                print(f"Unable to open given path or No Previous image present")

                tb = traceback.format_exc()

                gui.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)


            

    window.close()
if __name__ == "__main__":
    main()


