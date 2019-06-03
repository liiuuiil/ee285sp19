# ee285sp19 
[Neural Style Transfer](https://github.com/liiuuiil/ee285sp19/blob/master/README.md#neural-style-transfer)  [Cycle-GANs](https://github.com/liiuuiil/ee285sp19/blob/master/README.md#image-to-image-translation-using-cycle-gans)

This is Team LJSQ's ECE285 project. We choose project B - style transfer，including two parts, Neural Style Transfer and Image-to-Image Translation using Cycle-GANs.

## Neural Style Transfer

## Image-to-Image Translation using Cycle-GANs
### Prerequisites
  * Python 3.3 or above
  * Pytorch 0.4.0
  * Torchvision
### Quick satrt
    .ipynb combines test.py and will show you our result from trained model.
### Train
  'cd my-CycleGAN'
  'python train.py --display_id 0 --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan'
### Test
  'python test.py --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan'
