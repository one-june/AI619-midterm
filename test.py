#%%
import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as T

from vit import vit_small, CLSHead, DINOHead, SimpleWrapper

labels = ('nodule',)
ind_to_label = {i:label for i,label in enumerate(labels)}
label_to_ind = {v:k for k,v in ind_to_label.items()}

class GaussianBlurInference(object):
    """
    Apply Gaussian Blur to the PIL save.
    """
    def __init__(self):
        self.radius = 0.5

    def __call__(self, img):

        return img.filter(
            ImageFilter.GaussianBlur(self.radius)
            )

def load_img(img_path, img_size=(256,256), patch_size=8):
    img_path = str(img_path)
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    img = T.Compose(
        [
            GaussianBlurInference(),
            T.ToTensor()
        ]
    )(img) # ( 3, img_size[0], img_size[1] )
    
    # make the image divisible by patch size
    w, h = img.shape[1]-img.shape[1]%patch_size, img.shape[2]-img.shape[2]%patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    return img, w_featmap, h_featmap

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='checkpoint.pth')
    parser.add_argument('--testset_path', default='/media/wonjun/HDD2TB/VinDr-CXR-jpgs-resized512/test')
    return parser

def test(args):
    backbone = vit_small()
    dinohead = DINOHead()
    clshead = CLSHead()
    model = SimpleWrapper(backbone, dinohead, clshead)

    state_dict = torch.load(args.checkpoint_path)

    msg = model.load_state_dict(state_dict)
    print(msg)
    model.eval()
    model = model.to('cuda')

    testset_path = Path(args.testset_path)
    testdf = pd.read_csv('test.csv')

    ids = []
    preds = []
    for i, row in tqdm(testdf.iterrows(), total=len(testdf)):
        img_path = testset_path / f"{row['ID']}.jpg"
        img, _, _ = load_img(img_path)
        pred = model(img.to('cuda'))[0]
        pred = torch.sigmoid(pred).detach().cpu().numpy()[0][0]
        # pred = int(np.round(pred))
        pred = 1 if pred > 0.8 else 0
        
        ids.append(row['ID'])
        preds.append(pred)

    sample_submission = pd.DataFrame({'ID':ids, 'nodule':preds})
    sample_submission.to_csv('sample_submission.csv', index=False)

if __name__=="__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)


# %%
