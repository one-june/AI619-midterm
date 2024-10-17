#%%
import os
import cv2
import sys
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T

from vit import vit_small, CLSHead, DINOHead, SimpleWrapper
import utils


#%%
class NIHCXR_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 transforms,
                 ):
        self.transforms = transforms
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        # image = cv2.imread(img_path, 1) # 1 flag is for cv2.IMREAD_COLOR
        # image = cv2.resize(image, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        # image = Image.fromarray(image)
        image = Image.open(img_path).convert('RGB')
        
        images = self.transforms(image)
        
        label = torch.Tensor(row[['nodule']].values.astype(int))
        
        return images, label
#%%
def get_args_parser():
    parser = argparse.ArgumentParser('nodule-det', add_help=False)
    
    parser.add_argument('--trainset_csv', default='train.csv')
    parser.add_argument('--trainset_path', default='/media/wonjun/HDD2TB/')
    
    parser.add_argument('--save_path', default='checkpoint.pth')
    
    parser.add_argument('--batch_size_per_gpu', default=8)
    
    parser.add_argument('--lr', default=0.00005)
    parser.add_argument('--min_lr', default=1e-6)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--warmup_epochs', default=1)
    parser.add_argument('--weight_decay', default=0.01)
    parser.add_argument('--weight_decay_end', default=0.01)
    parser.add_argument('--momentum_teacher', default=0.9995)
    return parser

#%%
def train(args):
    df = pd.read_csv(args.trainset_csv)
    # df = df.head(50)
    df['img_path'] = df['img_path'].apply(lambda x: os.path.join(args.trainset_path, x))
    transforms = T.Compose(
        [
            T.Resize((224,224)),
            T.ToTensor()
        ]
    )

    dataset = NIHCXR_Dataset(df, transforms)
    sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    backbone = vit_small()
    dinohead = DINOHead()
    clshead = CLSHead()

    model = SimpleWrapper(backbone, dinohead, clshead)
    model = model.to('cuda')

    bce_loss = nn.BCEWithLogitsLoss()
    param_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(param_groups)
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu) / 16.,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )

    for epoch in range(args.epochs):
        
        epoch_loss = 0
        for it, (images, labels) in enumerate(data_loader):
            
            it = len(data_loader) * epoch + it  # global training iteration
            
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]
            
            images = images.cuda()
            labels = labels.float().cuda()
            
            preds = model(images)
            preds = torch.hstack(preds)
            loss = bce_loss(preds, labels)
            
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if (it+1) % 10 == 0:
            #     print(f"Epoch {epoch}, Global iteration {it+1}, loss {loss.item()}")
            
        print(f"Epoch {epoch}, loss {epoch_loss/len(data_loader)}")


    torch.save(model.state_dict(), args.save_path)

#%%
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)