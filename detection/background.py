import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import numpy as np




def load_background_backbone(input_shape=(224,224,3)):
    base_model=keras.applications.ResNet101(weights='imagenet',include_top=True,input_shape=input_shape)
    last_layer=base_model.layers[-2].output
    return keras.Model(base_model.input,last_layer)


def process_background(back_backbone,img):
    #img_background=preprocess(img)
    img=keras.preprocessing.image.img_to_array(img.resize((224,224)))
    img_background=keras.applications.resnet.preprocess_input(img)
        
    img_background=back_backbone(img_background[np.newaxis,...])
    return img_background

def combine_back_objs(im,objs,background_backbone):
    return np.concatenate([process_background(background_backbone,im),objs])






