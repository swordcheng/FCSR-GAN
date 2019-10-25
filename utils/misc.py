import os
import math
from PIL import Image
import numpy as np
import torch
import torchvision
# from skimage.measure import compare_ssim


def faceparsing2im(mask):

    mask = mask.detach()
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
               128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    mask = mask.reshape(-1, mask.shape[-1])
    mask = mask.unsqueeze_(2)
    mask = mask.cpu().float().numpy()
    mask = np.concatenate((mask, mask, mask), axis=2)

    new_mask = np.zeros(mask.shape)
    for i in range(11):
        flag = (mask == i) * 1.0
        color = np.array(palette[i * 3: i * 3 + 3])
        new_mask += color * flag

    new_mask = new_mask.astype(np.uint8)
    new_mask = Image.fromarray(new_mask)
    # new_mask = new_mask.resize((new_mask.size[0] * 2, new_mask.size[1] * 2))

    return np.array(new_mask).transpose((2, 0, 1))


def landmark2im(img):

    img = img.detach()
    img = torch.sum(img, dim=1, keepdim=True)
    img_ = torch.reshape(img, (img.shape[0], 1, -1))
    v_max, _ = torch.max(img_, dim=-1)
    img_ = img_ / v_max.unsqueeze_(2)
    img = torch.reshape(img_, img.shape)
    img = torch.reshape(img, (-1, img.shape[-1])).unsqueeze_(2)
    img = torch.cat((img, img, img), dim=2)
         
    img = img.cpu().float().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    
    img = Image.fromarray(img)
    img_shape = img.size

    img = img.resize((img_shape[0] * 2, img_shape[1] * 2))

    img = np.array(img).transpose((2, 0, 1))

    return img


def tensor2mask(img, idx=0, nrows=None, min_max=(0, 1)):

    if img.shape[1] == 1:
        img = torch.cat((img, img, img), dim=1)

    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows, padding=0)

    img = img.cpu().float()

    img = img.clamp_(*min_max)

    image_numpy = img.detach().numpy()
    image_numpy_t = image_numpy
    image_numpy_t = image_numpy_t*255

    return image_numpy_t.astype(np.uint8)


def tensor2im(img, unnormalize=True, idx=0, nrows=None, min_max=(0, 1)):

    if img.shape[1] == 1:
        img = torch.cat((img, img, img), dim=1)

    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows, padding=0)

    img = img.cpu().float()
    # if unnormalize:
    #     mean = [0.5, 0.5, 0.5]
    #     std = [0.5, 0.5, 0.5]
    #
    #     for i, m, s in zip(img, mean, std):
    #         i.mul_(s).add_(m)

    img = img.clamp_(*min_max)

    image_numpy = img.detach().numpy()
    image_numpy_t = image_numpy
    image_numpy_t = image_numpy_t*255

    return image_numpy_t.astype(np.uint8)


def mkdirs(paths):

    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):

    mkdir(os.path.dirname(image_path))

    image_pil = Image.fromarray(image_numpy)

    image_pil.save(image_path)


def save_str_data(data, path):

    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")


def dict_reader(file_list):

    img_dict = {}
    with open(file_list, 'r') as file:
        for line in file.readlines():
            ele = line[:-1].split()

            img_dict[ele[0]] = ele[1:]
    return img_dict


def list_reader(file_list):

    img_list = []
    with open(file_list, 'r') as file:
        for line in file.readlines():
            img_list.append(line[:-1].split()[0])
    return img_list


def loader(path):

    return Image.open(path)


def gaussian(size=3, sigma=0.25, amplitude=1, normalize=False,
             width=None, height=None, sigma_horz=None, sigma_vert=None,
             mean_horz=0.5, mean_vert=0.5):

    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)

    return gauss


def draw_gaussian(image, point, sigma):

    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1:
        return image
    size = 6 * sigma + 1
    g = gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1

    return image


def compute_loss_d(estim, is_real):

    return -torch.mean(estim) if is_real else torch.mean(estim)


def compute_loss_l1(feat1, feat2):

    l1_loss = torch.nn.L1Loss()

    return l1_loss(feat1, feat2)


def compute_loss_l2(feat1, feat2):

    mes_loss = torch.nn.MSELoss()

    return mes_loss(feat1, feat2)


def compute_loss_smooth(mat):

    return torch.mean(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
           torch.mean(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))


def compute_loss_gram_matrix(feat):

    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram


def compute_loss_cross_entropy(pred_parsing, gt_parsing):

    ce_loss = torch.nn.CrossEntropyLoss()
    return ce_loss(pred_parsing, gt_parsing.long())


def get_preds_fromhm(hm):

    _, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()

    preds[..., 0].add_(-1).fmod_(hm.size(3)).add_(1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            px, py = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if (0 < px < 63) and (0 < py < 63):
                diff = torch.cuda.FloatTensor(
                    [hm_[py, px + 1] - hm_[py, px - 1],
                     hm_[py + 1, px] - hm_[py - 1, px]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    return preds


def psnr(img1, img2):
    assert (img1.dtype == img2.dtype == np.uint8)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(im1, im2):
    # print(im1.shape, im2.shape)
    assert (im1.dtype == im2.dtype == np.uint8)
    assert (im1.ndim == im2.ndim == 2)

    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2
    l12 = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
    c12 = (2 * sigma1 * sigma2 + c2) / (sigma1 ** 2 + sigma2 ** 2 + c2)
    s12 = (sigma12 + c3) / (sigma1 * sigma2 + c3)

    return l12 * c12 * s12
