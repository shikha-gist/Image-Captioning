# LATGeO: Image-Captioning
This repository contains PyTorch implemetation of the paper [Label-Attention Transformer with Geometrically Coherent Objects for Image Captioning](https://arxiv.org/pdf/2109.07799.pdf) ! \
If you find our paper or provided codes helpful in your research, then please do not forget to cite our paper. Thank you! \
The following architecture represents our proposed model LATGeO for Image Captioning. \

## Requirements
<pre>
-python 3.8.8  
-pysimplegui 4.47.0
-pytorch 1.8.1
-torchvision  0.9.1
-numpy 1.18.5
-h5py 2.10.0 
-cython 0.29.23
-cudatoolkit 11.1.74
-pillow 8.2.0
-protobuf 3.17.3
-scipy 1.4.1
-tensorboard 2.4.0
-tensorflow-gpu 2.3.0
-spacy 3.0.6 
-python 3.8.8
-requests 2.24.0
-tqdm 4.60.0 
</pre>
**Detection using RCCN**: Follow the installation instructions provided by [Bottom-Up](https://github.com/peteanderson80/bottom-up-attention).
## Testing
The testing.py code is provided for predicting the detection of the provided image. \
We have also provided the Jupyter Notebook for better visualization of the predicted captions.\
Two Jupyter Notebooks are provided :
1. test_DETR-LATGeO.ipynb
2. test_RCNN-LATGeO.ipynb


## GUI-Demo
We have also provided the code for the GUI-Demo of our method.\
You could use the code for your application as well.\


## Citation
Please cite the following BibTex: 
<pre>
@misc{dubey2021labelattention, 
      title={Label-Attention Transformer with Geometrically Coherent Objects for Image Captioning}, 
      author={Shikha Dubey and Farrukh Olimov and Muhammad Aasim Rafique and Joonmo Kim and Moongu Jeon}, 
      year={2021}, 
      eprint={2109.07799}, 
      archivePrefix={arXiv}, 
      primaryClass={cs.CV} 
}
</pre>
If you find the paper and this repository helpful, please consider citing our paper [LATGeO](https://arxiv.org/pdf/2109.07799.pdf). Thank you!



## Acknowledgments
We would like to thanks [AImageLab](https://github.com/aimagelab/meshed-memory-transformer), [peteanderson80](https://github.com/peteanderson80/bottom-up-attention) and [facebookresearch](https://github.com/facebookresearch/detr) teams.
