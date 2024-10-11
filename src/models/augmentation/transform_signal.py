#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from multiprocessing.sharedctypes import Value
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


def torch_random_choice(val, size, weights=None):
    """
    Randomly sample a value from a list or tensor. replicates numpy.random.choice
    """
    if weights == None:
        weights = torch.tensor([1 / len(val)]).repeat(len(val))

    nb_sample = torch.tensor(size).prod()
    idx = torch.multinomial(weights, nb_sample, replacement=True)

    if type(val) != torch.Tensor:
        val = torch.Tensor([val])
    val_tile = val.repeat((nb_sample, 1))

    ret = torch.gather(val_tile, 1, torch.unsqueeze(idx, dim=0)).squeeze()
    ret = torch.reshape(ret, size)
    return ret


# All functions assume input x of size [batch, dim, time]
def jitter(x, sigma=0.1):
    noise_mag = x.std(dim=-1) * sigma
    noise_mag = noise_mag[:, :, None].repeat(1, 1, x.shape[-1])
    return x + torch.normal(torch.zeros_like(x), noise_mag)


def scaling(x, sigma=0.1):
    factor = torch.normal(1.0, sigma, size=(x.shape[0], x.shape[1]), device=x.device)
    # factor = torch.normal(1.0, sigma, size = (x.shape[0],), device=x.device)
    return torch.mul(x, factor[:, :, None])


def flip_y(x):
    flip = torch.randint(low=0, high=2, size=(x.shape[0], x.shape[1]), device=x.device)
    # flip = torch.randint(low=0, high=2, size=(x.shape[0],),device=x.device)
    flip[flip == 0] = -1  # reproduce random choice
    # rotate_axis =torch.arange(x.shape[1])

    return flip[:, :, None] * x


def shuffle_channel(x):
    rotate_axis = torch.arange(x.shape[1], device=x.device)
    rotate_axis = rotate_axis[torch.randperm(x.shape[1])]
    return x[:, rotate_axis, :]


def window_slice(x, reduce_ratio=0.8):
    target_len = torch.ceil(torch.tensor([reduce_ratio * x.shape[2]])).int()
    if target_len >= x.shape[2]:
        return x
    # starts = torch.randint(low=0, high=x.shape[2]-target_len[0], size=(x.shape[0],)).int()
    starts = torch.randint(low=0, high=x.shape[2] - target_len[0], size=(1,)).int()
    ends = (target_len + starts).int()
    ret = x[:, :, starts:ends]
    ret = F.interpolate(ret, size=x.shape[2], mode="nearest")

    return ret


def drop_block(x, drop_prob=0.1, block_size=7, scale=False):

    # get gamma value
    gamma = drop_prob / block_size
    gamma = gamma * (x.shape[-1] / (x.shape[-1] - block_size + 1)) *1.1

    # mask = torch.bernoulli(torch.ones((x.shape[0], *x.shape[2:])) * gamma)
    mask = torch.bernoulli(torch.ones((x.shape)) * gamma)

    # place mask on input device
    mask = mask.to(x.device)

    # compute block mask
    block_mask = F.max_pool1d(
        # input=mask[:, None, :],
        input=mask,
        kernel_size=block_size,
        stride=1,
        padding=block_size // 2,
    )

    if block_size % 2 == 0:
        block_mask = block_mask[:, :,:-1]

    block_mask = 1 - block_mask

    # apply block mask
    out = x * block_mask

    # scale output
    if scale:
        out = out * block_mask.numel() / block_mask.sum()

    return out


def random_block(x, drop_prob=0.1, block_size=7):
    "Replace random blocks of the input with normal noise"

    # get gamma value
    gamma = drop_prob / block_size
    gamma = (
        gamma * (x.shape[-1] / (x.shape[-1] - block_size + 1)) * 1.1
    )  # *1.2 to make sure we have enough points to drop
    # sample mask
    mask = torch.bernoulli(torch.ones((x.shape)) * gamma)

    # place mask on input device
    mask = mask.to(x.device)

    # compute block mask
    block_mask = F.max_pool1d(
        input=mask,
        kernel_size=(block_size),
        stride=(1),
        padding=block_size // 2,
    )

    if block_size % 2 == 0:
        block_mask = block_mask[:,:, :-1]

    block_mask = 1 - block_mask

    # apply block mask
    out = x * block_mask

    random_mask = torch.normal(mean=0, std=1, size=out.shape, device=x.device)
    random_mask = random_mask * (1 - block_mask)

    out += random_mask
    return out


def permute_block(x, drop_prob=0.1, block_size=7):
    "Replace random blocks of the input with permutation of the input signal"

    # get gamma value
    gamma = drop_prob / block_size
    gamma = (
        gamma * (x.shape[-1] / (x.shape[-1] - block_size + 1)) * 1.2
    )  # *1.2 to make sure we have enough points to drop

    # sample mask
    mask = torch.bernoulli(torch.ones((x.shape)) * gamma)

    # place mask on input device
    mask = mask.to(x.device)

    # compute block mask
    block_mask = F.max_pool1d(
        input=mask,
        kernel_size=(block_size),
        stride=(1),
        padding=block_size // 2,
    )

    if block_size % 2 == 0:
        block_mask = block_mask[:, :,:-1]

    block_mask = 1 - block_mask

    # apply block mask
    out = x * block_mask[:, None, :]

    indices = torch.argsort(torch.rand_like(x), dim=-1)
    random_permutation = torch.gather(x, dim=-1, index=indices)
    random_permutation = random_permutation * (1 - block_mask[:, None, :])

    out += random_permutation
    return out


def random_block_enforce_proba(x, drop_prob=0.1, block_size=7, scale=False):
    "Version of drop block with a while statement to enfore enough points are drop"

    # get gamma value
    gamma = drop_prob / block_size
    gamma = (
        gamma * (x.shape[-1] / (x.shape[-1] - block_size + 1)) * 1.2
    )  # *1.2 to make sure we have enough points to drop
    gamma = np.clip(gamma, 0, 1)
    diff_prob = torch.tensor(float("Inf"))

    count = 0
    while diff_prob > 0.05:
        mask = torch.bernoulli(torch.ones((x.shape)) * gamma)
        # sample mask

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = F.max_pool1d(
            # input=mask[:, None, :],
            input=mask,
            kernel_size=block_size,
            stride=1,
            padding=block_size // 2,
        )
        mask_prob = torch.sum(block_mask) / (x.shape[1]*x.shape[2])
        count += 1
        # print(count)
        # print("mask_prob", mask_prob)
        # print("drop_prob", drop_prob)
        # print("diff", abs(mask_prob - drop_prob))
        if abs(drop_prob - mask_prob) < diff_prob:
            diff_prob = abs(drop_prob - mask_prob)
            block_mask_retain = block_mask
        if count > 25:
            print(
                f"Could not find correct distribution to enforce drop proba in drop block, lowest diff prob {diff_prob}"
            )
            diff_prob = 0

    if block_size % 2 == 0:
        block_mask_retain = block_mask_retain[:,:, :-1]

    block_mask_retain = 1 - block_mask_retain

    # apply block mask to zero feature which are going to be replace with noise 
    out = x * block_mask_retain

    random_mask = torch.normal(mean=0, std=1, size=out.shape, device=x.device)
    random_mask = random_mask * (1 - block_mask_retain)

    out += random_mask

    return out

def random_block_shuffle(x, drop_prob, block_size):
    "Replace random blocks of the input with random shuffling of the block"
    gamma = drop_prob / block_size
    gamma = gamma * (x.shape[-1] / (x.shape[-1] - block_size + 1)) *1.1

    # mask = torch.bernoulli(torch.ones((x.shape[0], *x.shape[2:])) * gamma)
    mask = torch.bernoulli(torch.ones((x.shape[1:])) * gamma)[None, :]

    out = x.clone().detach()
    # x_out = x.copy()
    for c in range(mask.shape[1]):
        indices = torch.argwhere(mask[:,c,:])[:,-1]
        for idx in indices:
            idx_max = torch.min(torch.tensor((idx + block_size, x.shape[-1])))
            sample = out[:,c,idx:idx_max]
            idx_permute = torch.randperm(sample.shape[1])
            sample = sample[:,idx_permute]

            out[:,c,idx:idx_max] = sample
    
    return out