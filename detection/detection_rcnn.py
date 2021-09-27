import requests
import json
import numpy as np
from PIL import Image




labels_file='data/objects_vocab.txt'
classes = ['__background__']
with open(labels_file) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())
    classes=np.array(classes)


def get_detections(imageUrl):
    server_url=''  #provide your server URL for RCNN detector
    with open(imageUrl, 'rb') as f:
        file=f.read()
    resp = requests.request()  ### call RCNN detector here  
    output=resp.content
    output=json.loads(output)
    output={key:np.array(value) for key,value in output.items()}

    probs=output['cls_prob']
    bboxes=output['boxes']
    objs=output['features']

    labels=classes[probs.argmax(-1)]
    
    im=Image.open(imageUrl)
    return objs,bboxes,probs,labels, im
