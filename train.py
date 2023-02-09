import copy
import time
import os

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

import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from reid.pytorch_models import GoogleNet, ResNet18, ShuffleNet, ResNet, Net
from reid.reid_utils import l2_distance, triplet_loss
from argparse import ArgumentParser


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="res18",
        choices=["res18", "mobile", "google", "shuffle", "reid"],
        help="model name for training",
    )
    args.add_argument("--batch_size", type=int, default=32, help="batch size")
    args.add_argument("--epochs", type=int, default=50, help="number of epoch to train")
    args.add_argument(
        "--path", type=str, default="data/veri", help="path to test VeRi directory"
    )
    return args.parse_args()


parser = parse_args()
model_name = parser.model
batch_size = parser.batch_size
epochs = parser.epochs

NET_DICT = {
    "google": GoogleNet,
    "mobile": ResNet,
    "shuffle": ShuffleNet,
    "reid": Net,
    "res18": ResNet18,
}

model = NET_DICT[model_name]()
model.cuda()


CHECKPOINT_DIR = os.path.join(
    parser.path, "checkpoints", model_name
)  # f"checkpoints/{model_name}/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


for data_type in ["train", "test"]:
    img_ls = os.listdir(os.path.join(parser.path, f"image_{data_type}"))

    vec_id_list = []
    for img_ in img_ls:
        vec_id, *_ = img_.split("_")
        vec_id_list.append({"img_name": img_, "vehicleID": int(vec_id)})

    df = pd.DataFrame(vec_id_list)
    df.to_csv(os.path.join(parser.path, f"VeRi_{data_type}.csv"), index=False)

IMG_SHAPE = (128, 128, 3)
IMG_DIR = os.path.join(parser.path, "image_")

BATCH_SIZE = parser.batch_size  # 256
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


criterion = triplet_loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
steps_per_epoch = TRAINING_SAMPLES_NO // BATCH_SIZE


NUM_EPOCHS = parser.epochs  # 50

start = time.time()
for epoch in range(NUM_EPOCHS):
    start = time.time()
    avg_loss = 0.0

    model.train()
    for idx, (inp_imgs, labels) in enumerate(train_dataloader):
        inp_imgs, labels = inp_imgs.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = model(inp_imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.detach().cpu().numpy()
        print(f"\r Training Step : {idx} / {1000} Loss = {loss_val}", end="")
        avg_loss = (avg_loss * idx + loss_val) / (idx + 1)
        if idx > 1000:
            break

    scheduler.step()
    print(f"\r Epoch: {epoch} \t Training Loss: {avg_loss}")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, str(epoch))
    torch.save(model, checkpoint_path)

    model.eval()
    inp_imgs, labels = next(iter(test_dataloader))
stop = time.time()
print(stop - start)
