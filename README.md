# ee285sp19 
[Neural Style Transfer](https://github.com/liiuuiil/ee285sp19/blob/master/README.md#neural-style-transfer)  &   [Cycle-GANs](https://github.com/liiuuiil/ee285sp19/blob/master/README.md#image-to-image-translation-using-cycle-gans)

This is Team LJSQ's ECE285 project. We choose project B - style transfer，including two parts, Neural Style Transfer and Image-to-Image Translation using Cycle-GANs.

## Neural Style Transfer
### Prerequisites
  * Python 3.3 or above
  * Pytorch 0.4.0
  * Torchvision
### Getting Started
- Clone this repo:
```
git clone https://github.com/liiuuiil/ee285sp19
cd ee285sp19/nst
```
- Run neural style transfer
```
python nst.py 
```
- Example display
<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/nstoutput/house.png" />
<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/nstoutput/starrynight.png" />
<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/nstoutput/output.jpg" />
    
    
## Image-to-Image Translation using Cycle-GANs
### Prerequisites
  * Python 3.3 or above
  * Pytorch 0.4.0
  * Torchvision
### Quick satrt
    .ipynb combines test.py and will show you our result from trained model.
If you want to rerun the process, open GPU server terminal and do as follow
#### Train
    cd ./my-CycleGAN
    python train.py --display_id 0 --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan
#### Test
    python test.py --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan
### Reults
we focus on transferring from landscape photographs to two main artistic styles, Pointillism and Baroque.
   * Pointillism style

![image](https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_real_B.png)![image](https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_real_A.png)![image](https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_fake_A.png)

![image](https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch180_real_B.png)![image](https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch178_real_A.png)![image](https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch180_fake_A.png)
   * Baroque style
