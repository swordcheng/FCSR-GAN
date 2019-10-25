import os
import torch.utils.data
import numpy as np

from PIL import Image
from scipy.io import loadmat

import utils.misc as misc


class FaceImageLoader_w_aug(torch.utils.data.Dataset):

    def __init__(self, face_img_root, face_parsing_root, face_landmark_path,
                 face_list, scale_factor, img_size, heatmap_size, mode, upsample, transform=None):

        self.face_img_root = face_img_root
        self.face_parsing_root = face_parsing_root
        self.face_landmark_dict = misc.dict_reader(face_landmark_path)
        self.face_list = misc.list_reader(face_list)

        self.scale_factor = scale_factor
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.mode = mode
        self.upsample = upsample
        self.transform = transform

    def __getitem__(self, index):

        face_name = self.face_list[index]
        face_img = misc.loader(os.path.join(self.face_img_root, face_name))

        face_parsing = loadmat(os.path.join(self.face_parsing_root, face_name.split('.')[0] + '.mat'))['pos']

        face_landmark = self.face_landmark_dict[face_name]

        landmark_num = len(self.face_landmark_dict[face_name]) // 2

        point = np.zeros((landmark_num, 2))

        face_parsing = face_parsing - 1
        face_parsing = Image.fromarray(face_parsing.astype(np.uint8))

        heatmaps = np.zeros((landmark_num, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for i in range(landmark_num):
            point[i, 0] = point[i, 0] / 2
            point[i, 1] = point[i, 1] / 2
            heatmaps[i] = misc.draw_gaussian(heatmaps[i], point[i], 1)

        heatmaps = torch.from_numpy(heatmaps)

        idx = np.arange(0, self.img_size, self.img_size // self.heatmap_size)
        face_parsing = np.array(face_parsing)
        face_parsing = face_parsing[idx, :]
        face_parsing = face_parsing[:, idx]
        face_parsing = torch.from_numpy(face_parsing - 1)

        face_img_ = face_img.copy()

        face_img_ = face_img_.resize(
            (self.img_size // self.scale_factor, self.img_size // self.scale_factor), Image.BICUBIC)

        if self.upsample:
            face_img_ = face_img_.resize((self.img_size, self.img_size), Image.BICUBIC)

        mask = np.ones((face_img_.size[0], face_img_.size[1], 3)) * 255

        if self.mode == 'train':
            # top_x = np.random.randint(0, face_img_.size[0] // 2)
            # top_y = np.random.randint(0, face_img_.size[1] // 2)

            # patch_x = np.random.randint(10, face_img_.size[1] // 2)
            # patch_y = np.random.randint(10, face_img_.size[1] // 2)
            top_x = np.random.randint(0, face_img_.size[0] - 4)
            top_y = np.random.randint(0, face_img_.size[1] - 4)
            patch_x = np.random.randint(3, face_img_.size[0] - top_x)
            if patch_x > 16:
                patch_x = 16
            patch_y = np.random.randint(3, face_img_.size[1] - top_y)
            if patch_y > 16:
                patch_y = 16


        elif self.mode == 'val':

            top_x = face_img_.size[0] // 2
            top_y = face_img_.size[1] // 4
            patch_x = face_img_.size[0] // 4
            patch_y = face_img_.size[1] // 4

        else:
            raise ValueError("error!")

        mask[top_x:top_x + patch_x, top_y:top_y + patch_y, :] = 0

        face_img_masked = np.array(face_img_) * (mask / 255.0)

        face_img_masked = Image.fromarray(face_img_masked.astype(np.uint8))
        mask = torch.from_numpy(mask / 255).permute(2, 0, 1).float()

        point = torch.from_numpy(
            np.array([top_x, top_x + patch_x, top_y, top_y + patch_y]))

        # print(face_img.size, face_img_.size, face_img_masked.size)
        if self.transform is not None:
            face_img = self.transform(face_img)
            face_img_ = self.transform(face_img_)
            face_img_masked = self.transform(face_img_masked)

        # print(face_img.shape, face_img_.shape, face_img_masked.shape)

        sample = {
            'mask': mask,
            'img_LR_masked': face_img_masked,
            'img_LR': face_img_,
            'img_landmark': heatmaps,
            'img_face_parsing': face_parsing,
            'img_SR': face_img,
            'point': point
        }

        return sample

    def __len__(self):

        return len(self.face_list)

class FaceImageLoader(torch.utils.data.Dataset):

    def __init__(self, face_img_root, face_parsing_root, face_landmark_path,
                 face_list, scale_factor, img_size, heatmap_size, mode, upsample, transform=None):

        self.face_img_root = face_img_root
        self.face_parsing_root = face_parsing_root
        self.face_landmark_dict = misc.dict_reader(face_landmark_path)
        self.face_list = misc.list_reader(face_list)

        self.scale_factor = scale_factor
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.mode = mode
        self.upsample = upsample
        self.transform = transform

    def __getitem__(self, index):

        face_name = self.face_list[index]
        face_img = misc.loader(os.path.join(self.face_img_root, face_name))

        face_parsing = loadmat(os.path.join(self.face_parsing_root, face_name.split('.')[0] + '.mat'))['pos']

        face_landmark = self.face_landmark_dict[face_name]

        landmark_num = len(self.face_landmark_dict[face_name]) // 2

        point = np.zeros((landmark_num, 2))

        top_x = np.random.randint(0, 16)
        face_img = face_img.crop((top_x, top_x, 128, 128)).resize((128, 128), resample=Image.BICUBIC)

        for i in range(landmark_num):

            point[i, 0] = float(face_landmark[2 * i + 0]) - top_x
            point[i, 0] = point[i, 0] * 128 / (128 - top_x)
            point[i, 1] = float(face_landmark[2 * i + 1]) - top_x
            point[i, 1] = point[i, 1] * 128 / (128 - top_x)

        face_parsing = face_parsing - 1
        face_parsing = Image.fromarray(face_parsing.astype(np.uint8))
        face_parsing = face_parsing.crop((top_x, top_x, 128, 128)).resize((128, 128), resample=Image.BICUBIC)

        angle = np.random.randint(0, 30)
        face_parsing = face_parsing.rotate(angle)
        face_img = face_img.rotate(angle)
        for i in range(landmark_num):
            a = point[i, 0]
            b = point[i, 1]
            point[i, 0] = (a - 64) * np.cos(np.pi / 180 * -angle) - (b - 64) * np.sin(np.pi / 180 * -angle) + 64
            point[i, 1] = (a - 64) * np.sin(np.pi / 180 * -angle) + (b - 64) * np.cos(np.pi / 180 * -angle) + 64

        seed = np.random.randint(0, 2)
        if seed == 0:
            face_img = face_img.transpose(method=Image.FLIP_LEFT_RIGHT)
            face_parsing = face_parsing.transpose(method=Image.FLIP_LEFT_RIGHT)
            point[:, 0] = 128 - point[:, 0]
        # if seed == 1:
        #     face_img = face_img.transpose(method=Image.FLIP_TOP_BOTTOM)
        #     face_parsing = face_parsing.transpose(method=Image.FLIP_TOP_BOTTOM)
        #     point[:, 1] = 128 - point[:, 1]

        heatmaps = np.zeros((landmark_num, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for i in range(landmark_num):
            point[i, 0] = point[i, 0] / 2
            point[i, 1] = point[i, 1] / 2
            heatmaps[i] = misc.draw_gaussian(heatmaps[i], point[i], 1)

        heatmaps = torch.from_numpy(heatmaps)

        idx = np.arange(0, self.img_size, self.img_size // self.heatmap_size)
        face_parsing = np.array(face_parsing)
        face_parsing = face_parsing[idx, :]
        face_parsing = face_parsing[:, idx]
        face_parsing = torch.from_numpy(face_parsing - 1)

        face_img_ = face_img.copy()

        face_img_ = face_img_.resize(
            (self.img_size // self.scale_factor, self.img_size // self.scale_factor), Image.BICUBIC)

        if self.upsample:
            face_img_ = face_img_.resize((self.img_size, self.img_size), Image.BICUBIC)

        mask = np.ones((face_img_.size[0], face_img_.size[1], 3)) * 255

        if self.mode == 'train':
            
            top_x = np.random.randint(0, face_img_.size[0] - 4)
            top_y = np.random.randint(0, face_img_.size[1] - 4)
            patch_x = np.random.randint(3, face_img_.size[0] - top_x)
            if patch_x > 16:
                patch_x = 16
            patch_y = np.random.randint(3, face_img_.size[1] - top_y)
            if patch_y > 16:
                patch_y = 16


        elif self.mode == 'val':

            top_x = face_img_.size[0] // 4
            top_y = face_img_.size[1] // 4
            patch_x = face_img_.size[0] // 2
            patch_y = face_img_.size[1] // 2
            
        else:
            raise ValueError("error!")

        mask[top_x:top_x + patch_x, top_y:top_y + patch_y, :] = 0

        face_img_masked = np.array(face_img_) * (mask / 255.0)

        face_img_masked = Image.fromarray(face_img_masked.astype(np.uint8))
        mask = torch.from_numpy(mask / 255).permute(2, 0, 1).float()

        point = torch.from_numpy(
            np.array([top_x, top_x + patch_x, top_y, top_y + patch_y]))

        # print(face_img.size, face_img_.size, face_img_masked.size)
        if self.transform is not None:
            face_img = self.transform(face_img)
            face_img_ = self.transform(face_img_)
            face_img_masked = self.transform(face_img_masked)

        # print(face_img.shape, face_img_.shape, face_img_masked.shape)

        sample = {
            'mask': mask,
            'img_LR_masked': face_img_masked,
            'img_LR': face_img_,
            'img_landmark': heatmaps,
            'img_face_parsing': face_parsing,
            'img_SR': face_img,
            'point': point
        }

        return sample

    def __len__(self):

        return len(self.face_list)
