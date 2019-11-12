from torch.utils.data import *
from imutils import paths
import cv2
import numpy as np
from os.path import isfile, join
import os
from PIL import Image


def load_path(img_dir):
  files = os.listdir(img_dir)
  files = list(filter(lambda a: a[-3:] == 'txt', files))
  img_paths = []
  boxes = []
  lp_nums = []
  for f in files:
    fp= open(join(img_dir, f), "r")
    text = fp.read()
    content = text.strip().split('\t')
    img_paths.append(content[0])
    boxes.append(content[1:5])
    lp_nums.append(content[-1])
  return img_paths[:20], boxes[:20], lp_nums[:20]

class lpDataLoader(Dataset):
  def __init__(self, img_dir, input_shape, ads, model_name, is_transform=None):
    self.img_dir = img_dir
    self.ads = ads
    self.img_paths, self.boxes, self.lp_nums = load_path(self.img_dir)
    self.input_shape = input_shape
    self.model_name = model_name
    self.is_transform = is_transform

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, index):
    img_name = self.img_paths[index]
    image = Image.open(join(self.img_dir, img_name))
    iw, ih = image.size
    box =  np.array([int(i) for i in self.boxes[index]])
    lp_num = self.lp_nums[index]
    lp_num = [int(self.ads[lp_num[i]]) for i in range(len(lp_num))]
    for i in range(10 - len(lp_num)):
      lp_num.append(0)
    h, w = self.input_shape

    box[0] = box[0] - (box[2] / 2)
    box[1] = box[1] - (box[3] / 2)
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]

    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image)/255.
    image_data = image_data.astype('float32')
    image_data = np.transpose(image_data, (2,0,1))

    box[[0,2]] = box[[0,2]] * scale + dx
    box[[1,3]] = box[[1,3]] * scale + dy
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    cx = box[0] + (bw / 2)
    cy = box[1] + (bh / 2)

    new_label = [cx / w, cy / h, bw / w, bh / h]
    if self.model_name == 'lpnet':
      return image_data, new_label, lp_num, img_name
    else:
      return image_data, new_label
