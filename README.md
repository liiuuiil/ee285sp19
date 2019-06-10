# ee285sp19 
[Neural Style Transfer](https://github.com/liiuuiil/ee285sp19/blob/master/README.md#neural-style-transfer)  &   [Cycle-GANs](https://github.com/liiuuiil/ee285sp19/blob/master/README.md#image-to-image-translation-using-cycle-gans)

This is Team LJSQ's ECE285 project. We choose project B - Style Transfer, including two parts, Neural Style Transfer and Image-to-Image Translation using Cycle-GANs.

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

<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/nstoutput/house.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/nstoutput/starrynight.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/nstoutput/output.png" width="200"/>
 
    
## Image-to-Image Translation using Cycle-GANs
### Prerequisites
  * Python 3.3 or above
  * Pytorch 0.4.0
  * Torchvision
### Quick start
    demo.ipynb contains test.py and will show you all the results.
If you want to rerun the process, open GPU server terminal and do as follow.
Download or git clone our project to your GPU server.

We include checkpoints file in our project. Since the original checkpoints is over 8G, the current checkpoints only contain the latest training model and loss we get. We have also attached the link of checkpoints at https://drive.google.com/file/d/1gWMMvxNYSieGRgjfeb4VQfyWxWMoK_Qy/view?usp=sharing using your UCSD email, then save checkpoints in your file path.

Do not change any name of file, in case of meeting any accidental problem when using the following commands.
#### Train
    cd ./ dir of our file 'ee285sp19'
    python train.py --display_id 0 --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan
#### Test
    cd ./ dir of our file 'ee285sp19'(if you skip train step)
    python test.py --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan
### Results
    check test results at the newly produced 'cycleGANs_outputs' file
we focus on transferring from landscape photographs to two artistic styles, Pointillism and Abstract Expressionism style.
   * Pointillism style
   
   
<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch158_real_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch158_real_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch158_fake_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch158_fake_B.png" width="200"/>

<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_real_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_real_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_fake_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch198_fake_B.png" width="200"/>

<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch180_real_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch180_real_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch180_fake_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Pointillism%20style/epoch180_fake_B.png" width="200"/>

   * Abstract Expressionism style
   
   
<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/179_real_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/179_real_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/179_fake_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/179_fake_A.png" width="200"/>

<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/136_real_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/136_real_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/136_fake_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/136_fake_A.png" width="200"/>

<img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/256_real_A.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/256_real_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/256_fake_B.png" width="200"/><img src="https://github.com/liiuuiil/ee285sp19/blob/master/image/Abstract%20Expressionism%20style/256_fake_A.png" width="200"/>
