#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F 
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision as tv


# In[93]:


device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[94]:


def getimage(image_path,image_size=(512,512)):
    transform = tv.transforms.Compose([
            tv.transforms.Resize(image_size),
            tv.transforms.ToTensor(),
            ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img[None,:,:,:]
    return img.to(device, torch.float)


# In[95]:


def myimshow(image, title,save=False,ax=plt):
    fig=plt.figure()  
    image = image.squeeze(0) 
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) 
    h = ax.imshow(image) 
    ax.title(title)
    ax.axis('off') 
    
    if save:
        fig.savefig('{}.jpg'.format(title))
    return h


# In[96]:


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


# In[97]:


def gram_matrix(x):
    a, b, c, d = x.size()  
    features = x.view(a * b, c * d)  
    G = torch.mm(features, features.t())/(a*b*c*d)  
    return G


# In[98]:


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


# In[99]:


def getmodel(style_img, content_img,content_layers,style_layers):
    
    vgg = tv.models.vgg19(pretrained=True).features.to(device)
    model = nn.Sequential()

    i = 0  
    j = 0
    m = 0
    n = 0
    for layer in vgg:
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'Conv_{}'.format(i)
        else:
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            j += 1
            name = 'Notconv_{}'.format(j)

        model.add_module(name, layer)

        if name[5:] in str(content_layers):
            # add content loss:
            m += 1
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(m), content_loss)
          

        if name[5:] in str(style_layers):
            # add style loss:
            n += 1
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(n), style_loss)
            
        if m==len(content_layers) and n==len(style_layers):
            break
    return model


# In[107]:


def style_transfer(input_img,content_img, style_img,
                       content_layers,style_layers,
                       style_weight=1000000, content_weight=1,T=2000,lr=0.001,):
    
    model = getmodel(style_img, content_img,content_layers,style_layers)
    optimizer = torch.optim.Adam([input_img.requires_grad_()],lr=lr)

    for epoch in range(T):
        input_img.data.clamp_(0, 1)
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
        
        if epoch % 200 == 199:
            print("run {}:".format(epoch+1))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))

    input_img.data.clamp_(0, 1)
    return input_img


# In[108]:


style_img = getimage('starry.jpg')
content_img = getimage('house.jpg')
myimshow(style_img,'style_img')
myimshow(content_img,'content_img')


# In[109]:


content_layers_0 = [4]
style_layers_0 = [1,2,3,4]
style_layers_1= [1]
style_layers_2= [1,2]
style_layers_3= [1,2,3]


# In[110]:


input_img = content_img.clone()


# In[111]:


output = style_transfer(input_img,content_img, style_img,content_layers=content_layers_0,style_layers=style_layers_0,)


# In[112]:


myimshow(output.detach(),'output',save=True)


# In[ ]:





# In[ ]:




