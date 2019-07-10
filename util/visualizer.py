"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from . import util
from tensorboardX import SummaryWriter

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        self.writer = SummaryWriter(self.log_dir)

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        visuals = self.convert_visuals_to_numpy(visuals)
                
        for label, image_numpy in visuals.items():
            self.writer.add_image(label, image_numpy, epoch)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        for tag, value in errors.items():
            value = value.mean().float()
            self.writer.add_scalar('loss/' + tag, value, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t)
                if len(t.shape) == 4:
                    t = t[0]
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
