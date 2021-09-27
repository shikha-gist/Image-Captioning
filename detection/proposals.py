import torch
import numpy as np
import detection.detection_detr as detr
import detection.detection_rcnn as rcnn
from PIL import Image
import detection.background as background
import re
import matplotlib.pyplot as plt



obj_backbone=detr.load_obj_backbone()
background_backbone=background.load_background_backbone()

def plot_results(pil_img, prob, boxes,CLASSES):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = detr.COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        if p[p.argmax()]<0.7:
            continue
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
   

def plot_results_GUI(pil_img, prob, boxes,CLASSES):   ### for GUI 
    plt.figure(figsize=(16,10))

    plt.imshow(pil_img)
    ax = plt.gca()
    colors = detr.COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        if p[p.argmax()]<0.7:
            continue
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')    
    plt.savefig('temp.png', pad_inches = 0, bbox_inches='tight', transparent=True)
    plt.close()
    im_url=Image.open('temp.png')
    return im_url


def detr_detection(detrModel,imageUrl,device,show_objs):  
    
    im = Image.open(imageUrl)
    #
    img = detr.transform(im).unsqueeze(0)
    size=torch.tensor(im.size,device=device)
    # propagate through the model
    outputs = detrModel(img.to(device))

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
    keep = probas.max(-1).values > 0.5

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = detr.rescale_bboxes(outputs['pred_boxes'][0, keep], size).cpu().numpy()

    # extract  Classses to obj based on probs 
    labels=detr.CLASSES[probas[keep].cpu().numpy().argmax(-1)] 
    labels=['background',*labels]
    
    # extract objs  from image by cutting    

    objs=[im.crop(bbox).resize((224,224)) for bbox in bboxes_scaled]
    if show_objs==True:
        
        plot_results(im,probas[keep],bboxes_scaled,detr.CLASSES)
    ## add probs for backgoround clolumns-wise first ,and then row-wise
    probs=probas[keep].cpu().numpy()

    
   ## 
    probs=np.concatenate([np.zeros((probs.shape[0],1)),probs],axis=-1) 
    probs=np.concatenate([np.zeros((1,probs.shape[1])),probs],axis=0)
    probs[0,0]=1.0
    probs=np.max(probs,axis=-1)
    
    ## Add the bbox for background 

    #
    bboxes_scaled=np.concatenate([[[0,0,*im.size]],bboxes_scaled],axis=0)

    
    return objs,bboxes_scaled,probs,labels, im

def rccn_detection(imageUrl,show_objs):
    objs,bboxes,probs,labels, im=rcnn.get_detections(imageUrl)
   
    if show_objs==True:
        plot_results(im,probs,bboxes,rcnn.classes)
        
    ## adding background probs
    probs=np.concatenate([np.zeros((probs.shape[0],1)),probs],axis=-1) 
    probs=np.concatenate([np.zeros((1,probs.shape[1])),probs],axis=0)
    probs[0,0]=1.0
    probs=np.max(probs,axis=-1)
  
    ##  
    bboxes=np.concatenate([[[0,0,*im.size]],bboxes],axis=0)


    labels=['scene',*labels]
    return objs,bboxes,probs,labels, im

def stadndardize(features ,bboxes,probas,labels,text_field,max_detections=50):
    delta = max_detections - features.shape[0]  ##till here
    if delta > 0:
        features = np.concatenate([features, np.zeros((delta, features.shape[1]))], axis=0)
        bboxes=np.concatenate([bboxes,np.zeros((delta,bboxes.shape[1]))],axis=0)
        labels=np.concatenate([labels,np.repeat(['<pad>'],delta)],axis=-1)
        probas=np.concatenate([probas,np.zeros(delta)],axis=-1)
    elif delta < 0:
        features = features[:max_detections]
        bboxes=bboxes[:max_detections]
        labels=labels[:max_detections]
        probas=probas[:max_detections]
    #precomp_labels=np.array([text_field.numericalize([[re.sub('(\W+\w+)','',col )]])for col in labels]) 
    precomp_labels=np.array([text_field.numericalize([[re.sub('([ ,.\-\']+\w+)','',col )]])for col in labels]) 
    features=torch.Tensor(features).unsqueeze(0)
    bboxes=torch.Tensor(bboxes).unsqueeze(0)
    precomp_labels=torch.Tensor(precomp_labels).unsqueeze(0).int()
    probas=torch.Tensor(probas).unsqueeze(0)
    return features ,bboxes,probas,precomp_labels
    

def process_image(imageUrl,detectionModel,text_field,args,max_detections=50,device=torch.device('cuda:0'),show_objs=True):
   if args.detection=='detr':
       objs,bbox,probs,labels,im=detr_detection(detectionModel,imageUrl,device,show_objs)
       objs=detr.process_objs(obj_backbone,objs)
   elif args.detection=='rcnn': ## here if args.detection is not detr , then detection Model is ignored!
      objs,bbox,probs,labels,im= rccn_detection(imageUrl,show_objs)

   objs=background.combine_back_objs(im,objs,background_backbone)
   features ,bboxes,probas,precomp_labels= stadndardize(objs,bbox,probs,labels,text_field,max_detections)
   return features ,bboxes,probas,precomp_labels,im

   


   

    
    
