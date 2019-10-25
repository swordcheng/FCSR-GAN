import numpy as np
import os
import time
from . import misc
from PIL import Image
# from tensorboardX import SummaryWriter
# from PIL import Image


class TBVisualizer:
    def __init__(self, opt):

        self._opt = opt
        self._save_path = os.path.join(opt.checkpoints_dir, opt.name)

        self._log_path = os.path.join(self._save_path, 'loss_log.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        # self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # def __del__(self):
    #     self._writer.close()

    def display_current_results(self, visuals, i_epoch, it, is_train, save_visuals):

        img_concat = None

        for label, image_numpy in visuals.items():

            image = Image.fromarray(image_numpy.transpose((1, 2, 0)))
            scale_factor = 128 // image.size[0]
            image = image.resize((scale_factor * image.size[0], scale_factor * image.size[1]), Image.BICUBIC)
            image_numpy = np.array(image)

            if img_concat is not None:
                img_concat = np.concatenate((img_concat, image_numpy), axis=1)
            else:
                img_concat = image_numpy
        
        sum_name = 'Train_' if is_train else 'Test_'
        sum_name = sum_name + str('%03d' % i_epoch)
        # self._writer.add_image(sum_name, image_numpy, it)
        if save_visuals:
            misc.save_image(img_concat,
                            os.path.join(self._opt.checkpoints_dir, self._opt.name,
                                         'event_imgs-' + sum_name, '%08d' % it + '-' + sum_name + '.png'))

        # self._writer.export_scalars_to_json(self._tb_path)

    def plot_scalars(self, scalars, it, is_train):

        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            # self._writer.add_scalar(sum_name, scalar, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):

        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.6fs) ' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.6f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):

        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, visuals):
        
        for label, image_numpy in visuals.items():
            image_name = '%s.png' % label
            save_path = os.path.join(self._save_path, "samples", image_name)
            misc.save_image(image_numpy, save_path)
