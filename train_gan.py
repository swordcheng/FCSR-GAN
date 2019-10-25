import os
import time
import numpy as np
import torch.utils.data as torch_data

from PIL import Image
from torchvision import transforms

from data.my_data_loader import FaceImageLoader
from models.base_models import ModelFactory
from utils.tb_visualizer import TBVisualizer
from options.train_options_model_combine_pconv_fsrnet import TrainOptions

import utils.misc as misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Train:

    def __init__(self):

        self._opt = TrainOptions().parse()
        self.face_train_list = self._opt.face_train_list
        self.face_test_list = self._opt.face_test_list

        self.face_img_root = self._opt.face_img_root
        self.face_parsing_root = self._opt.face_parsing_root
        self.face_landmark_train = self._opt.face_landmark_train
        self.face_landmark_test = self._opt.face_landmark_test

        self.img_size = self._opt.img_size
        self.scale_factor = self._opt.scale_factor
        self.img_size = self._opt.img_size
        self.heatmap_size = self._opt.heatmap_size

        self.scale_factor = self._opt.scale_factor

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5],
            #                      std=[0.5, 0.5, 0.5])
        ])

        self._dataset_train = torch_data.DataLoader(dataset=FaceImageLoader(
            self._opt.face_img_root,
            self._opt.face_parsing_root,
            self._opt.face_landmark_train,
            self.face_train_list,
            transform=self.train_transform,
            scale_factor=self.scale_factor,
            img_size=self.img_size,
            heatmap_size=self.heatmap_size,
            mode='train',
            upsample=self._opt.upsample),
            batch_size=self._opt.batch_size,
            num_workers=self._opt.n_threads_train,
            shuffle=True, drop_last=True)

        self._dataset_test = torch_data.DataLoader(dataset=FaceImageLoader(
            self._opt.face_img_root,
            self._opt.face_parsing_root,
            self._opt.face_landmark_test,
            self.face_test_list,
            transform=self.test_transform,
            scale_factor=self.scale_factor,
            img_size=self.img_size,
            heatmap_size=self.heatmap_size,
            mode='val',
            upsample=self._opt.upsample),
            batch_size=10,
            num_workers=self._opt.n_threads_test,
            shuffle=False, drop_last=True)

        self._dataset_train_size = len(self._dataset_train)
        self._dataset_test_size = len(self._dataset_test)

        print('#train images = %d' % self._dataset_train_size)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelFactory.get_by_name(self._opt.model, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)

        self._save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        self._val_path = os.path.join(self._save_path, 'val_log.txt')


        self._train()

    def _train(self):

        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):

            if i_epoch > 1:
                epoch_val_start_time = time.time()
                self._val_epoch(i_epoch)
                epoch_val_end_time = time.time()
                time_epoch = epoch_val_end_time - epoch_val_start_time
                print('End of epoch %d / %d \t Val Time Taken: %d sec (%d min or %d h)' %
                      (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                       time_epoch / 60.0, time_epoch / 3600.0))

            epoch_start_time = time.time()
            self._train_epoch(i_epoch)
            print('saving the model at the end of epoch %d, iters %d' % (i_epoch, self._total_steps))
            self._model.save(i_epoch)
            time_epoch = time.time() - epoch_start_time

            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs_no_decay + self._opt.nepochs_decay, time_epoch,
                   time_epoch / 60.0, time_epoch / 3600.0))

            if i_epoch > self._opt.nepochs_no_decay:
                self._model.update_learning_rate()

    def _val_epoch(self, i_epoch):

        self._model.set_eval()

        for i_val_batch, val_batch in enumerate(self._dataset_test):

            point = val_batch['point'].numpy()

            self._model.set_input(val_batch)
            self._model.forward(keep_data_for_visuals=True)
            visuals = self._model.get_current_visuals()

            img_sr = visuals['batch_img_fine'].transpose((1, 2, 0))
            img_gt = visuals['batch_img_SR'].transpose((1, 2, 0))

            self._tb_visualizer.display_current_results(visuals,
                                                        i_epoch,
                                                        i_val_batch,
                                                        is_train=False,
                                                        save_visuals=True)
        self._model.set_train()

    def _train_epoch(self, i_epoch):

        epoch_iter = 0
        self._model.set_train()

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_start_time = time.time()

            do_visuals = (self._last_display_time is None) or \
                         (time.time() - self._last_display_time > self._opt.display_freq_s)
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s or do_visuals

            self._model.set_input(train_batch)
            train_generator = ((i_train_batch + 1) % self._opt.train_G_every_n_iterations == 0) or do_visuals
            self._model.optimize_parameters(train_generator, keep_data_for_visuals=do_visuals)

            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, do_visuals)
                self._last_print_time = time.time()

            if do_visuals:
                self._display_visualizer_train(i_epoch, self._total_steps)
                self._last_display_time = time.time()

            if self._last_save_latest_time is None or \
                    time.time() - self._last_save_latest_time > self._opt.save_latest_freq_s:
                print('saving the latest model (epoch %d, total_steps %d)' % (i_epoch, self._total_steps))
                self._model.save(i_epoch)
                self._last_save_latest_time = time.time()

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, visuals_flag):

        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch,
                                                       self._iters_per_epoch, errors, t, visuals_flag)

    def _display_visualizer_train(self, i_epoch, total_steps):

        self._tb_visualizer.display_current_results(self._model.get_current_visuals(),
                                                    i_epoch,
                                                    total_steps,
                                                    is_train=True,
                                                    save_visuals=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_errors(), total_steps, is_train=True)
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(), total_steps, is_train=True)


if __name__ == "__main__":
    Train()
