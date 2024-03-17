import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Additional Scripts
#from config import cfg


class DentalDataset(Dataset):
    #output_size = cfg.transunet.img_dim

    def __init__(self, path, transform):
        super().__init__()

        self.transform = transform

        img_folder = os.path.join(path, 'img')
        mask_folder = os.path.join(path, 'mask')

        self.img_paths = []
        self.mask_paths = []
        for p in os.listdir(img_folder):
            name = p.split('.')[0]

            self.img_paths.append(os.path.join(img_folder, name + '.npy'))
            self.mask_paths.append(os.path.join(mask_folder, name + '.npy'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.load(self.img_paths[idx])
        mask = np.load(self.mask_paths[idx])

        sample = {'img': img, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        img, mask = sample['img'], sample['mask']
        img = torch.from_numpy(img.astype('float32')).permute((2, 0, 1))
        img = img.to(dtype=torch.float)
        mask = torch.from_numpy(mask)
        mask = F.one_hot(mask.to(torch.int64), num_classes=5).permute((2, 0, 1))
        mask = mask.to(dtype=torch.long)
        
        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    #import torchvision.transforms as transforms
    #from utils import transforms as T

    # transform = transforms.Compose([T.BGR2RGB(),
    #                                 T.Rescale(cfg.input_size),
    #                                 T.RandomAugmentation(2),
    #                                 T.Normalize(),
    #                                 T.ToTensor()])

    md = DentalDataset('/home/graciela/Adrian/TransUnet/transunet_pytorch/trainset',
                       None) #'/home/kara/Downloads/UFBA_UESC_DENTAL_IMAGES_DEEP/dataset_and_code/test/set/train'

    for sample in md:
        print(sample['img'].shape)
        print(sample['mask'].shape)
        '''cv2.imshow('img', sample['img'])
        cv2.imshow('mask', sample['mask'])
        cv2.waitKey()'''

        break
