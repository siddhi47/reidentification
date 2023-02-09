import torch
import torch.nn.functional as F


def l2_distance(a, b=None):
    b = a if b is None else b
    a_sq = torch.sum(torch.square(a), dim=1)
    b_sq = torch.sum(torch.square(b), dim=1)
    dist_sq = a_sq.view((-1, 1)) + b_sq.view((1, -1)) - 2 * a @ b.T
    return torch.sqrt(torch.maximum(torch.zeros_like(dist_sq), 1e-5 + dist_sq))


def triplet_loss(embed, labels):
    dist_matrix = l2_distance(embed)
    label_matrix = (
        torch.eq(torch.unsqueeze(labels, dim=1), torch.unsqueeze(labels, dim=0)) * 1.0
    )
    max_pos_dist = torch.amax(dist_matrix * label_matrix, dim=1)
    min_neg_dist = torch.amin(dist_matrix + (label_matrix * 1e10), dim=1)
    loss = F.softplus(max_pos_dist - min_neg_dist)
    return torch.mean(loss)
