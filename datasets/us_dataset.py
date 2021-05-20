from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
SIZE_W = 256
SIZE_H = 256


class AortaDataset(Dataset):
    def __init__(self, hparams, datasets):
        self.device = hparams.device
        self.n_classes = hparams.n_classes
        print('hparams.data_root: ', hparams.data_root)

        self.resize = True
        self.scale = 1
        self.mask_suffix = ''
        assert 0 < self.scale <= 1, 'Scale must be between 0 and 1'

        self.imgs_ids, self.masks_ids = ([] for i in range(2))
        img_type = ["/imgs/", "/masks/"]
        for fold in datasets:
            dir = hparams.data_root + "/" + fold
            [self.imgs_ids.append(dir + img_type[0] + file) for file in listdir(dir + img_type[0])]
            [self.masks_ids.append(dir + img_type[1] + file) for file in listdir(dir + img_type[1])]

        logging.info(f'Creating dataset with {len(self.imgs_ids)} examples')

    def __len__(self):
        return len(self.imgs_ids)

    def add_dataset_specific_args(parser):
        parser.add_argument("--testing_this", type=float, default=1)

    @classmethod
    def change_size(cls, pil_img, resize, scale):
        if resize:
            pil_img = pil_img.resize((SIZE_W, SIZE_H))
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        return pil_img

    @classmethod
    def preprocess(cls, pil_img, mask):
        img_nd = np.array(pil_img)
        if mask:
            img_nd = np.where(img_nd == 2, 0, img_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img = Image.open(self.imgs_ids[i])
        mask = Image.open(self.masks_ids[i])

        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img = self.change_size(img, self.resize, self.scale)
        mask = self.change_size(mask, self.resize, self.scale)

        img = self.preprocess(img, mask=False)
        mask = self.preprocess(mask, mask=True)

        return torch.from_numpy(img).type(torch.FloatTensor).to(self.device), \
               torch.from_numpy(mask).type(torch.FloatTensor).to(self.device), \
               self.imgs_ids[i]

    def add_dataset_specific_args(parser):  # pragma: no cover
        specific_args = parser.add_argument_group(title='database specific args options')
        specific_args.add("--dimensions", default=1, type=int)
        return parser
