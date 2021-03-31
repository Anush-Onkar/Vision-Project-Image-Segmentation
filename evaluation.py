#Project Submission by Anush Onkarappa (7010620- anon00001@stud.uni-saarland.de) & Hitesh Kotte (7010571-hiko00001@stud.uni-saarland.de)
import torchvision
from torchvision import torch 
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import random
from random import shuffle
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import torch
import torch.nn as nn
from R2_Unet_model import *
from Metrics import *


#Normalization and Transformation for dataset

#root_path = './Cityscape'
root_path=''
data_train=torchvision.datasets.Cityscapes(root=root_path,mode='fine',target_type='semantic', 
                                           transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                                           [0.229, 0.224, 0.225]), transforms.Resize((512,256))]), 
                                           target_transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,256))]))
# dataloader variable
trainloader = torch.utils.data.DataLoader(data_train,batch_size=1,shuffle=True)

CUDA = torch.cuda.is_available()
torch.cuda.set_device(7)

if CUDA:
    print("CUDA is available")
    model = torch.load('model.pt')
    model.cuda()
    
else:
    
    print("CUDA not available, model will be on CPU")
    model = torch.load('model.pt')

epoch=2
dir= ""
for epoch in range(epochs): 
    running_loss = 0.0
        
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs= (inputs*255).cuda()
        labels= (labels*255).long().reshape(labels.size()[0],512,256).cuda()
        # forward + backward + optimize
        outputs = model.forward(inputs)
        loss = loss_f(outputs, labels)
        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 iteration
            print('epoch{}, iter{}, loss: {}'.format(epoch,i,loss.data))






f1_score,sensitivity,jaccard_score,accuracy=evaluate(labels,outputs)

print("Accuracy: " + str(round((accuracy*100), 2)) + "%")
print("F1 Score: " + str(f1_score*100) + "%")
print("Sensitivity: " + str(round((sensitivity*100), 2)) + "%")
print("Jaccard Score: " + str(round((jaccard_score*100), 2)) + "%")