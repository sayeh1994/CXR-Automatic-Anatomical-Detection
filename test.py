'''
python DETR/test.py --img_org="Path/to/image"\\ 
'''
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
from torch import nn

from torchvision.models import resnet50
import torchvision.transforms as T
from models import build_model

import argparse
from arguments import get_args_parser

import random

torch.set_grad_enabled(False);
CLASSES = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone', 'right hilar structures', 
           'right apical zone','right costophrenic angle', 'right cardiophrenic angle','right hemidiaphragm',
           'left lung','left upper lung zone','left mid lung zone','left lower lung zone','left hilar structures',
           'left apical zone','left costophrenic angle','left hemidiaphragm','trachea','spine','right clavicle',
           'left clavicle','aortic arch','mediastinum','upper mediastinum','svc','cardiac silhouette',
           'left cardiac silhouette','right cardiac silhouette','cavoatrial junction','right atrium','descending aorta',
           'carina','left upper abdomen','right upper abdomen','abdomen','left cardiophrenic angle']

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def read_image(image_path):
    return Image.open(image_path).convert('RGB'), cv2.imread(image_path)

def main(args):
    
    model, criterion, postprocessors = build_model(args)
    state_dict = torch.load(args.read_checkpoint)
    model.load_state_dict(state_dict["model"])
    model.eval()
    
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_org_pl, image_org_cv = read_image(args.img_org)
    image = image_org_cv.copy()
    scores_org, boxes_org = detect(image_org_pl, model, transform)
    
    colors = [(random.randint(0, 100), random.randint(0, 100), 160) for _ in range(len(boxes_org))]
    for p, (x1, y1, w, h), color in zip(scores_org, boxes_org.tolist(), colors):
        print([x1,y1,w,h])

        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.rectangle(image, (int(x1), int(y1)), (int(w), int(h)), color, thickness=5)
        cv2.putText(image, text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    cv2.imwrite("AnDetectCXR",image)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
