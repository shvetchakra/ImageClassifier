import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
import argparse
import os
import numpy as np
from PIL import Image
import itertools

parser1 = argparse.ArgumentParser()
parser1.add_argument('img_path')
parser1.add_argument('checkpoint')
parser1.add_argument('--top_k', type = int, default = 1)
parser1.add_argument('--category_names', type = str, default = 'cat_to_name.json')
parser1.add_argument('--gpu', action='store_true')

args1 = parser1.parse_args()
img_path = args1.img_path
checkpoint = args1.checkpoint
topk = args1.top_k
category_names = args1.category_names
gpu = args1.gpu
device = 'cpu'
if gpu :
    device = 'cuda'
cat_to_name = {}
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
def load_checkpoints(filepath):
    checkpoint = torch.load(filepath,map_location='cuda:0')
    model = get_model(checkpoint['model_name'])
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'],checkpoint['hidden_units'] ),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']),
                                         nn.LogSoftmax(dim=1))
    model.optimizer = checkpoint['optimizer_state']
    model.criterion = checkpoint['criterion']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def get_model(model_name='densenet121'):
    if model_name.islower() == "vgg13":
        return models.vgg13(pretrained=True)
    if model_name.islower() == "alexnet":
        return models.alexnet(pretrained=True)
    if model_name.islower() == "squeezenet1_0":
        return models.squeezenet1_0(pretrained=True)
    if model_name.islower() == "resnet18":
        return models.resnet18(pretrained=True)
    return models.densenet121(pretrained=True)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 256, 256
    pil_image=image.resize(size)
    pil_image = pil_image.crop(((256-224)//2, (256-224)//2, (256+224)//2, (256+224)//2))
    np_image = np.array(pil_image,dtype=np.double)/255
  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    
    return torch.from_numpy(np_image.T)
def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    im = Image.open(image_path)
    tensor_image = process_image(im)
    inputs = tensor_image.unsqueeze(0)
    #model=load_checkpoint('checkpoint3.pth')
    #device = 'cpu'
    inputs = inputs.float()
    inputs.to(device)
    
    model.to(device)
    with torch.no_grad():
        model.eval()
        
        logps = model.forward(inputs)
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class

model = get_model()
model = load_checkpoints(checkpoint)
top_p, top_class = predict(img_path,model)
def oneDArray(x):
    return list(itertools.chain(*x))
l = top_class.tolist()
ll = oneDArray(l)
classes=[]
for i in ll:
    classes.append(cat_to_name[str(i)])
top_p = top_p.tolist()
top_p = oneDArray(top_p)
for i in range(len(top_p)):
    print(f"Probability: {(top_p[i]):.3f}.. "
	f"Class Category: {classes[i]}")