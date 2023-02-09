import copy
import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from PIL import Image
from torchsummary import summary
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from PIL import Image
from torch.utils.data import DataLoader
from argparse import ArgumentParser

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

from reid.pytorch_models import GoogleNet, ResNet18, ShuffleNet, ResNet, Net
from reid.reid_utils import l2_distance, triplet_loss

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    arg_parse = ArgumentParser()
    arg_parse.add_argument(
        "--model",
        type=str,
        default="res18",
        choices=["res18", "mobile", "google", "shuffle", "reid"],
        help="model name",
    )
    arg_parse.add_argument("--batch_size", type=int, default=64, help="batch size")
    arg_parse.add_argument("--epoch", type=int, default=49, help="epoch to load")
    arg_parse.add_argument(
        "--path", type=str, default="VeRi/", help="path to VeRi folder"
    )
    return arg_parse.parse_args()


parser = parse_args()
model_name = parser.model

NET_DICT = {
    "google": GoogleNet,
    "mobile": ResNet,
    "shuffle": ShuffleNet,
    "reid": Net,
    "res18": ResNet18,
}

model = NET_DICT[model_name]()
model.cuda()

for data_type in ["train", "test"]:
    img_ls = os.listdir(os.path.join(parser.path, f"image_{data_type}"))

    vec_id_list = []
    for img_ in img_ls:
        vec_id, *_ = img_.split("_")
        vec_id_list.append({"img_name": img_, "vehicleID": int(vec_id)})

    df = pd.DataFrame(vec_id_list)

    df.to_csv(os.path.join(parser.path, f"VeRi_{data_type}.csv"), index=None)

IMG_SHAPE = (128, 128, 3)
IMG_DIR = os.path.join(parser.path, "image_")

BATCH_SIZE = parser.batch_size
TRAINING_SAMPLES_NO = 4000

BATCH_NO_IMAGES_PER_VEHICLE = 4
BATCH_NO_VEHICLE = BATCH_SIZE // BATCH_NO_IMAGES_PER_VEHICLE


data_transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SHAPE[:-1]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class VeRi_Dataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_file, data_type="train"):
        super(VeRi_Dataset).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.Resize(IMG_SHAPE[:-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        df = pd.read_csv(csv_file)
        self.unique_vehicles = df.vehicleID.unique()

        self.IMG_DIR = IMG_DIR + (
            "train" if data_type in ["train", "valid"] else "test"
        )

        if data_type == "train":
            self.unique_vehicles = self.unique_vehicles[:-20]
        elif data_type == "valid":
            self.unique_vehicles = self.unique_vehicles[-20:]

        self.df = df[df.vehicleID.apply(lambda x: x in self.unique_vehicles)]

    def __iter__(self):
        while True:
            batch = []
            batch_vehicles = np.random.choice(
                self.unique_vehicles, size=BATCH_NO_VEHICLE, replace=False
            )

            for vehicle_id in batch_vehicles:
                df_sample = self.df[self.df.vehicleID == vehicle_id].sample(
                    BATCH_NO_IMAGES_PER_VEHICLE
                )
                batch.append(df_sample)

            df_sample = pd.concat(batch)

            for idx in range(len(df_sample)):
                filename, label = df_sample.iloc[idx]

                img_path = os.path.join(self.IMG_DIR, filename)
                img = Image.open(img_path)
                yield self.transforms(img), label

    def __len__(self):
        return len(self.df)


train_dataset = VeRi_Dataset(os.path.join(parser.path, "VeRi_train.csv"), "train")
valid_dataset = VeRi_Dataset(os.path.join(parser.path, "VeRi_train.csv"), "valid")
test_dataset = VeRi_Dataset(os.path.join(parser.path, "VeRi_test.csv"), "test")

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE * 2, num_workers=0, shuffle=False
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE * 2, num_workers=0, shuffle=False
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, num_workers=0, shuffle=False
)


def calculate_model_accuracy(model, test_dataloader, threshold):
    acc = 0
    model.eval()

    for idx, (inp_imgs, labels) in enumerate(test_dataloader):
        dist_mat = l2_distance(model(inp_imgs.cuda())).cpu()
        pred_ids = (dist_mat < threshold) * 1

        label_matrix = (
            torch.eq(torch.unsqueeze(labels, dim=1), torch.unsqueeze(labels, dim=0))
            * 1.0
        )
        # plt.imshow((pred_ids))
        # plt.show()

        acc_ = (label_matrix == pred_ids).sum() / (
            len(label_matrix) * len(label_matrix)
        )
        acc = (acc * idx + acc_) / (idx + 1)

        if idx > test_comb:
            return acc


def calculate_fps(model, single_img_inp):
    time_ls = []

    for i in range(5000):
        start = time.time()
        model(single_img_inp.cuda())
        stop = time.time()

        time_taken = stop - start
        time_ls.append(time_taken)
    fps = 1 / np.median(time_ls)
    return fps


CHECKPOINT_DIR = os.path.join(parser.path, f"checkpoints/{model_name}/{parser.epoch}")
print("Loading model")
model = torch.load(CHECKPOINT_DIR)
test_comb = len(test_dataloader.dataset) // BATCH_SIZE
threshold = 0.8

acc = calculate_model_accuracy(model, test_dataloader, threshold)
print(f"Final Accuracy: {acc:.3f}")
inp_imgs, _ = next(iter(test_dataloader))
single_img_inp = inp_imgs[:1]
fps = calculate_fps(model, single_img_inp)
print(f"FPS: {fps:.3f}")

with open("acc_fps.txt", "a") as f:
    f.write(
        f"model: {model_name} epoch: {parser.epoch} accuracy: {acc:.3f} fps: {fps:.3f}\n"
    )
