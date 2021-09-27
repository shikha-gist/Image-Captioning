import random 
from caption_model.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory,GeomtericAttention
import torch
import numpy as np
from torch import nn
import multiprocessing

def load_capModel(args,text_field,device):
    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=GeomtericAttention,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

   

    fname = args.load_weight
     


    data = torch.load(fname)
    model.load_state_dict(data['state_dict'], strict=False)

    label_state={}
    def_states=['decoder.label_attention.in_proj_weight', 'decoder.label_attention.in_proj_bias', 'decoder.label_attention.out_proj.weight', 'decoder.label_attention.out_proj.bias']

    for key,value in data['state_dict'].items():
        if key in def_states:
            label_state['.'.join(key.split('.')[1:])]=value
        
    model.decoder.label_attention.load_state_dict(label_state)
   
    


    print('Loading from epoch %d, validation loss %f, and best cider %f' % (
               data['epoch'], data['val_loss'], data['best_cider']))

    return model
