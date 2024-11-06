


# 在这个文件中，我们创建了一个简单的模型，然后对模型进行简单的inference，测试整个运行的pipeline


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import pickle

from main import get_args_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()#有默认值
    print(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(args.device)
    # print('model.device:',model.device)
    
    #读取输入数据和标签
    with open('samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    with open('targets.pkl', 'rb') as f:
        targets = pickle.load(f)

    samples = samples.to(args.device)
    outputs = model(samples)
    # print('outputs ',outputs)
    # print('samples ',samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
    loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                for k, v in loss_dict_reduced.items() if k in weight_dict}
    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

    loss_value = losses_reduced_scaled.item()
    print('ok!!!!!!!!!!!!!!!!!!!!!!')