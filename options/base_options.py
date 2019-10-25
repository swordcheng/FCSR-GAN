import argparse
import os
from utils import misc


class BaseOptions():

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser.add_argument('--face_train_list', type=str, default='/home/jccai/ssd/celebA/train.txt', help='')
        
        self._parser.add_argument('--face_test_list', type=str, default='/home/jccai/ssd/celebA/test.txt', help='')

        self._parser.add_argument('--face_img_root', type=str, default='/home/jccai/ssd/celebA/img_/', help='')
        
        self._parser.add_argument('--face_parsing_root', type=str,
                                  default='/home/jccai/ssd/celebA/face_parsing_mat/', help='')
        self._parser.add_argument('--face_landmark_train', type=str,
                                  default='/home/jccai/ssd/celebA/face_landmark_train.txt', help='')
        self._parser.add_argument('--face_landmark_test', type=str,
                                  default='/home/jccai/ssd/celebA/face_landmark_test.txt', help='')

        self._parser.add_argument('--image_size', type=int, default=128, help='input image size')
        self._parser.add_argument('--gpu_ids', type=str, default='0',
                                  help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self._parser.add_argument('--checkpoints_dir', type=str, default='../FCSRGAN-saved/checkpoints', help='')
        self._initialized = True

    def parse(self):

        if not self._initialized:
            self.initialize()

        self._opt = self._parser.parse_args()

        self._opt.is_train = self.is_train

        self._set_and_check_load_epoch()

        self._get_set_gpus()

        args = vars(self._opt)

        self._print(args)

        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):

        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        # print(int(file.split('_')[2]))
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):

        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

    def _print(self, args):

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        misc.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
