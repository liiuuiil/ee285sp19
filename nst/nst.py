#!/usr/bin/env python
# coding: utf-8

# In[238]:


import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F 
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision as tv


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def getimage(image_path,image_size=(512,512)):
    transform = tv.transforms.Compose([
            tv.transforms.Resize(image_size),
            tv.transforms.ToTensor(),
            ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img[None,:,:,:]
    return img.to(device, torch.float)

#starry.jpg and house.jpg should be replaced by the path of the image chosen.
style_img = getimage('starry.jpg') 
content_img = getimage('house.jpg')

def myimshow(image, title,ax=plt):
    plt.figure()  
    image = image.squeeze(0) 
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) 
    h = ax.imshow(image) 
    ax.title(title)
    ax.axis('off') 
    return h

myimshow(style_img,'style_img')
myimshow(content_img,'content_img')

class ContentLoss(nn.Module):

    def __init__(self, f_content,):
        super(ContentLoss, self).__init__()
        self.f_content = f_content.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.f_content)
        return x

def gram_matrix(x):
    a, b, c, d = x.size()  

    features = x.view(a * b, c * d)  

    G = torch.mm(features, features.t())  
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, f_style):
        super(StyleLoss, self).__init__()
        self.f_style = gram_matrix(f_style).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.f_style)
        return x


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_layers_1= ['conv_1']
style_layers_2= ['conv_1','conv_2']
style_layers_3= ['conv_1','conv_2','conv_3']
style_layers_4= ['conv_1','conv_2','conv_3','conv_4']


# In[248]:


def getmodel(style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    
    vgg = tv.models.vgg19(pretrained=True).features.to(device)
    model = nn.Sequential()
    i = 0  
    j = 0
    m = 0
    n = 0
    for layer in vgg:
        if isinstance(layer, nn.Conv2d):
            i += 1
            j = 0
            name = 'conv_{}'.format(i)
        else:
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            j += 1
            name = 'sublayer_{}'.format(j)

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            m += 1
            f_content = model(content_img).detach()
            content_loss = ContentLoss(f_content)
            model.add_module("content_loss_{}".format(m), content_loss)
    

        if name in style_layers:
            # add style loss:
            n += 1
            f_style = model(style_img).detach()
            style_loss = StyleLoss(f_style)
            model.add_module("style_loss_{}".format(n), style_loss)
            
        if m==len(content_layers) and n==len(style_layers):
            break
    return model


# In[249]:


input_img = content_img.clone()


# In[250]:


def style_transfer(input_img,content_img, style_img, 
                       content_layers=content_layers_default,style_layers=style_layers_default,
                       style_weight=1000000, content_weight=1,T=1000,gamma=.001,rho=0.9,tol=20):

    model= getmodel(style_img, content_img,content_layers,style_layers)
    optimizer = torch.optim.SGD([input_img.requires_grad_()], lr=gamma, momentum=rho)


    for epoch in range(T):

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for name, module in model.named_children():
            if 'style_loss_' in name:
                style_score += module.loss
            if 'content_loss' in name:
                content_score += module.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        

        optimizer.step()
        if epoch % 100 == 0:
            print("epoch {}:".format(epoch))
            print('Style Loss : {:f} Content Loss: {:f} Total Loss {:f}'.format(
                style_score.item(), content_score.item(),loss.item()))

        if loss.item()<tol:
            break
    
    return input_img




output = style_transfer(input_img,content_img, style_img)




myimshow(output.detach(),'output')
