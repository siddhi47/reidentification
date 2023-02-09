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

from reid.pytorch_models import GoogleNet, ResNet, ShuffleNet, Net, ResNet18
from reid.reid_utils import l2_distance, triplet_loss

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "--model",
        type=str,
        default="res18",
        choices=["res18", "mobile", "google", "shuffle", "reid"],
        help="model name",
    )
    args.add_argument("--epoch", type=int, default=49, help="Which epoch to load")
    args.add_argument("--path", type=str, default="VeRi", help="path to VeRi folder")
    parser = args.parse_args()
    return parser


parser = parse_args()


class VeriDataset(data.Dataset):
    def __init__(self, data_dir, train_list, train_data_transform=None, is_train=True):
        super(VeriDataset, self).__init__()

        self.is_train = is_train
        self.data_dir = data_dir
        self.train_data_transform = train_data_transform
        reader = open(train_list)
        lines = reader.readlines()
        self.names = []
        self.labels = []
        self.cams = []
        if is_train:
            for line in lines:
                line = line.strip().split(" ")
                self.names.append(line[0])
                self.labels.append(line[1])
                self.cams.append(line[0].strip().split("_")[1])
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split("_")[0])
                self.cams.append(line.split("_")[1])

    def __getitem__(self, index):
        # For normalize

        img = Image.open(os.path.join(self.data_dir, self.names[index])).convert(
            "RGB"
        )  # convert gray to rgb
        target = int(self.labels[index])
        camid = self.cams[index]

        if self.train_data_transform:
            img = self.train_data_transform(img)

        return img, target, camid

    def __len__(self):
        return len(self.names)


def get_dataset(query_dir, query_list, gallery_dir, gallery_list):
    data_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    query_set = VeriDataset(query_dir, query_list, data_transform, is_train=False)
    gallery_set = VeriDataset(gallery_dir, gallery_list, data_transform, is_train=False)

    query_loader = DataLoader(
        dataset=query_set, num_workers=4, batch_size=1, shuffle=False
    )
    gallery_loader = DataLoader(
        dataset=gallery_set, num_workers=4, batch_size=1, shuffle=False
    )

    return query_loader, gallery_loader


def evaluate(query_loader, gallery_loader, model):
    print("Start evaluation...")
    query_feats = []
    query_pids = []
    query_camids = []

    gallery_feats = []
    gallery_pids = []
    gallery_camids = []

    end = time.time()
    # switch to eval mode
    model.eval()

    print("Processing query set...")
    queryN = 0
    for i, (image, pid, camid) in enumerate(query_loader):
        # if i == 10:
        #     break
        # print('Extracting feature of image ' + '%d:' % i)
        query_pids.append(pid)
        query_camids.append(camid)
        image = torch.autograd.Variable(image).cuda()
        feat = model(image)
        query_feats.append(feat.data.cpu())
        queryN = queryN + 1

    query_time = time.time() - end
    end = time.time()
    print("Processing query set... \tTime[{0:.3f}]".format(query_time))

    print("Processing gallery set...")
    galleryN = 0
    for i, (image, pid, camid) in enumerate(gallery_loader):
        # if i == 20:
        #     break
        # print('Extracting feature of image ' + '%d:' % i)
        gallery_pids.append(pid)
        gallery_camids.append(camid)
        image = torch.autograd.Variable(image).cuda()
        feat = model(image)
        gallery_feats.append(feat.data.cpu())
        galleryN = galleryN + 1

    gallery_time = time.time() - end
    print("Processing gallery set... \tTime[{0:.3f}]".format(gallery_time))
    print("Computing CMC and mAP...")
    cmc, mAP, distmat = compute(
        query_feats,
        query_pids,
        query_camids,
        gallery_feats,
        gallery_pids,
        gallery_camids,
    )
    print("Saving distmat...")
    np.save(save_dir + "distmat.npy", np.asarray(distmat))
    np.savetxt(save_dir + "distmat.txt", np.asarray(distmat), fmt="%.4f")
    print("mAP = " + "%.4f" % mAP + "\tRank-1 = " + "%.4f" % cmc[0])

    return mAP


def compute(
    query_feats, query_pids, query_camids, gallery_feats, gallery_pids, gallery_camids
):
    # query
    qf = torch.cat(query_feats, dim=0)

    q_pids = np.array(query_pids)
    q_camids = np.array(query_camids).T

    # gallery
    gf = torch.cat(gallery_feats, dim=0)
    g_pids = np.array(gallery_pids)
    g_camids = np.array(gallery_camids).T

    m, n = qf.shape[0], gf.shape[0]
    qf = qf.view(m, -1)
    gf = gf.view(n, -1)
    print("Saving feature mat...")
    np.save(save_dir + "queryFeat.npy", qf)
    np.save(save_dir + "galleryFeat.npy", gf)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()

    q_camids = np.squeeze(q_camids)
    g_camids = np.squeeze(g_camids)

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP, distmat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    # max_rank = TopK
    TopK = max_rank
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print("Saving resulting indexes...", indices.shape)
    np.save(save_dir + "result.npy", indices[:, :TopK] + 1)
    np.savetxt(save_dir + "result.txt", indices[:, :TopK] + 1, fmt="%d")

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


NET_DICT = {
    "google": GoogleNet,
    "mobile": ResNet,
    "shuffle": ShuffleNet,
    "reid": Net,
    "res18": ResNet18,
}

model = NET_DICT[parser.model]()
model.cuda()


# Create dataloader
print("====> Creating dataloader...")

query_dir = os.path.join(parser.path, "image_query")
query_list = os.path.join(parser.path, "list/veri_query_list.txt")
gallery_dir = os.path.join(parser.path, "image_test")
gallery_list = os.path.join(parser.path, "list/veri_test_list.txt")

query_loader, gallery_loader = get_dataset(
    query_dir, query_list, gallery_dir, gallery_list
)
save_dir = "save_dir/"
mkdir_if_missing(save_dir)
CHECKPOINT_DIR = os.path.join(parser.path, f"checkpoints/{parser.model}/{parser.epoch}")
model = torch.load(CHECKPOINT_DIR)
cudnn.benchmark = True
mAP = evaluate(query_loader, gallery_loader, model)
with open("map.txt", "a") as f:
    f.write(f"model: {parser.model} epoch: {parser.epoch} mAP : {mAP}\n")
