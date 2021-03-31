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
from Atten_R2Unet_model import *
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
    model = R2AttU_Net(img_ch=3,output_ch=34,t=2).cuda()
    
else:
    
    print("CUDA not available, model will be on CPU")
    model = R2AttU_Net(img_ch=3,output_ch=34,t=2)

print(model)



bs =8   #change later 
epochs = 2  #change later 
learning_rate = 0.0001  #change later 


# loss function- CrossEntropy 
loss_f = nn.CrossEntropyLoss()

# optimizer variable
# using Adam as optimizer 
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)



epoch=2
#Model training 
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
        loss.backward()
        opt.step()

        # zero the parameter gradients
        opt.zero_grad()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 iteration
            print('epoch{}, iter{}, loss: {}'.format(epoch,i,loss.data))
    #To save model with epoch
    #torch.save(model.state_dict(), os.path.join(dir, 'epoch-{}.pt'.format(epoch)))
    #save whole model         
    torch.save(model,os.path.join(dir,'model.pt' ))  
print('Finished Training')
