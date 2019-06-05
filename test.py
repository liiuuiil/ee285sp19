"""CycleGAN model for image-to-image translation.
    python test.py --dataroot ./datasets/285cyclegan --name 285cyclegan_cyclegan --model cycle_gan
"""
import os
import ntpath
from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util.util import tensor2im
from util.util import save_image
from scipy.misc import imresize

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):

    images, words, links = [], [], []
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    image_dir = webpage.get_image_dir()
    webpage.add_header(name)

    for tag, image in visuals.items():
        im = tensor2im(image)
        image_name = '%s_%s.png' % (name, tag)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        save_image(im, save_path)

        images.append(image_name)
        words.append(tag)
        links.append(image_name)
    webpage.add_images(images, words, links, width=width)


if __name__ == '__main__':
    opt = TestOptions().parse()
    
#     # test for demo
#     opt.name = 'face_cyclegan'
#     opt.dataroot = './datasets/face'
#     opt.model = 'cycle_gan'
    
    # parameters     
    opt.num_threads = 0  
    opt.batch_size = 1    
    opt.serial_batches = True 
    opt.no_flip = True   
    opt.display_id = -1  
    dataset = create_dataset(opt)  
    model = create_model(opt)      
    model.setup(opt)      
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  
            break
        model.set_input(data) 
        model.test()         
        visuals = model.get_current_visuals()  
        img_path = model.get_image_paths()   
        if i % 5 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  

