import json
import random
import os

import open_clip
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger


def read_files(dir_path):
    all_data = None
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # print(file_path)
            caps = np.load(file_path)
            if all_data is None:
                all_data = caps
            else:
                all_data = np.concatenate((all_data, caps), axis=0)
    print(all_data.shape)
    return all_data


def count_trainable_parameters(model):
    """To compute the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """To compute the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def set_random_seed(seed):
    """To set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def load_data(args):
    datas = []
    text_dir = f"data/text_pair_data/{args.dataset}"
    image_dir = f"data/image_data/{args.dataset}/"
    for root, dirs, files in os.walk(text_dir):
        for file_name in files:
            if file_name.find("-") != -1:
                file_path = os.path.join(root, file_name)
                file = np.load(file_path)
                for item in file:
                    # print(image_dir,item[0])
                    datas.append([image_dir + item[0], item[1]])
    return datas


def init_model(args):
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model, pretrained=args.pretrained_model
    )

    model.to(args.device)

    logger.info("model parameters: {}".format(count_trainable_parameters(model)))

    tokenizer = open_clip.get_tokenizer(args.model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr if "lr" in args else 1e-4
    )

    criterion = open_clip.CoCaLoss(
        caption_loss_weight=args.caption_loss_weight,
        clip_loss_weight=1,
    )

    return model, transform, tokenizer, optimizer, criterion
