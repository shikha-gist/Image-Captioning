import torch
import utils
from caption_model.containers import Module
from caption_model.beam_search import *


class CaptioningModel(Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    
    def beam_search(self, visual: utils.TensorOrSequence,boxes:utils.TensorOrSequence, labels:utils.TensorOrSequence,probs:utils.TensorOrSequence,max_len: int, eos_idx: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(visual,boxes,labels ,probs,out_size, return_probs, **kwargs)
