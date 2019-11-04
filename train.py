from lpnet import lpnet
from load_data import *

from torch.utils.data import *
from imutils import paths
from os.path import isfile, join
import cv2
import os
import numpy as np

numPoints = 4
numClasses = 35
batchSize = 16
imgSize = (480, 480)

def load_path(img_dir):
    files = os.listdir(img_dir)
    files = list(filter(lambda a: a[-3:] == 'txt', files))
    img_paths = []
    labels = []
    for f in files:
        fp= open(join(img_dir, f), "r")
        text = fp.read()
        content = text.strip().split('\t')
        img_paths.append(content[0])
        labels.append(content[1:])
    return img_paths, labels

class LpLocDataLoader(Dataset):
    def __init__(self, img_dir,imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths, self.labels = load_path(self.img_dir)
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(join(self.img_dir, img_name))
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        #assert img.shape[0] == 1160
        label = self.labels[index]
        new_labels = [(int(label[0]) + int(label[2]))/(2*ori_w), (int(label[1]) + int(label[3]))/(2*ori_h), (int(label[2])-int(label[0]))/ori_w, (int(label[3])-int(label[1]))/ori_h]

        resizedImage = resizedImage.astype('float32')
        # Y = Y.astype('int8')
        resizedImage /= 255.0
        # lbl = img_name.split('.')[0].rsplit('-',1)[-1].split('_')[:-1]
        # lbl = img_name.split('/')[-1].split('.')[0].rsplit('-',1)[-1]
        # lbl = map(int, lbl)
        # lbl2 = [[el] for el in lbl]

        # resizedImage = torch.from_numpy(resizedImage).float()
        return resizedImage, new_labels

lpnet = lpnet(numPoints, numClasses, model_name= 'base', net_file= None, base_file = None)
dst = LpLocDataLoader('img', imgSize)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)
lpnet.train(trainloader, batchSize, epoch_start= 0, epochs= 25)
