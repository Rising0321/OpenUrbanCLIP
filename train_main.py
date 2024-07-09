import math
import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from data.datasets import CoCaDataset
from utils.utils import (
    count_trainable_parameters,
    set_random_seed,
    unwrap_model,
    maybe_compute_generative_loss,
    get_clip_metrics,
    read_files, load_data, init_model,
)
from tqdm import tqdm
from loguru import logger

def create_datasets(args, transform, tokenizer):
    data = load_data(args)

    np.random.shuffle(data)

    train_dataset_ratio = 0.7
    val_dataset_ratio = 0.15
    test_dataset_ratio = 0.2

    train_data = data[: int(len(data) * train_dataset_ratio)]
    val_data = data[int(len(data) * train_dataset_ratio): int(len(data) * (train_dataset_ratio + val_dataset_ratio))]
    test_data = data[int(len(data) * (train_dataset_ratio + val_dataset_ratio)):]

    # create datasets
    train_dataset = CoCaDataset(train_data, transform, tokenizer, args.dataset)
    val_dataset = CoCaDataset(val_data, transform, tokenizer, args.dataset)
    test_dataset = CoCaDataset(test_data, transform, tokenizer, args.dataset)

    train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    data = {}
    data["train_loader"] = train_dataset
    data["val_loader"] = val_dataset
    data["test_loader"] = test_dataset

    return data


def train_one_epoch(model, criterion, data, epoch, optimizer, args, device):
    model.train()

    dataloader = data["train_loader"]

    progress_bar = tqdm(dataloader)

    progress_bar.set_description(f"Train Epoch {epoch}")

    for batch in dataloader:
        optimizer.zero_grad()

        images, texts = batch
        images = images.to(device=device)
        texts = texts.to(device=device)

        if texts.ndim == 3:
            texts = texts.squeeze(1)

        model_out = model(images, texts)
        losses = criterion(**model_out, output_dict=True)
        total_loss = sum(losses.values())
        losses["loss"] = total_loss

        total_loss.backward()
        optimizer.step()
        logs = {"loss": total_loss.item()}
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)


def eval(model, criterion, data, epoch, optimizer, args, device, phase):
    with torch.no_grad():
        model.eval()

        dataloader = data[phase + "_loader"]

        progress_bar = tqdm(dataloader)

        progress_bar.set_description(f"{phase} Epoch {epoch}")

        res = []
        for batch in dataloader:
            images, texts = batch
            images = images.to(device=device)
            texts = texts.to(device=device)

            if texts.ndim == 3:
                texts = texts.squeeze(1)

            model_out = model(images, texts)
            losses = criterion(**model_out, output_dict=True)
            total_loss = sum(losses.values())
            losses["loss"] = total_loss

            logs = {"loss": total_loss.item()}
            progress_bar.set_postfix(**logs)
            res.append(logs["loss"])

            progress_bar.update(1)

        return {"clip_val_loss": np.mean(res)}


def init(args):
    set_random_seed(args.seed)
    # create logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./checkpoints/upstream"):
        os.makedirs("./checkpoints/upstream")

    logger.remove(handler_id=None)  # remove default logger
    logger.add(os.path.join("./logs", str(args.seed) + ".log"), level="INFO")
    logger.info(args)


def eval_best(model, args, data, criterion, epoch, optimizer):
    best_checkpoint = torch.load(
        os.path.join("./checkpoints/upstream", "best_model.bin"),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(best_checkpoint)
    model.to(args.device)
    test_metric = eval(model, criterion, data, epoch, optimizer, args, args.device, "val")
    logger.info("test metric: {}".format(test_metric))



def main(args):
    init(args)

    model, transform, tokenizer, optimizer, criterion = init_model(args)

    # create datasets
    data = create_datasets(
        args, transform, tokenizer
    )

    save_path = f"./checkpoints/upstream/{args.model}-{args.name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_clip_val_loss = float("inf")

    for epoch in range(args.epoch_num):
        logger.info("Start epoch {}".format(epoch))
        print("Start epoch {}".format(epoch))
        train_one_epoch(model, criterion, data, epoch, optimizer, args, args.device)
        completed_epoch = epoch + 1

        cur_metrics = eval(model, criterion, data, epoch, optimizer, args, args.device, "val")

        if cur_metrics["clip_val_loss"] < best_clip_val_loss:
            print("--------------------enter saving mode--------------------")
            checkpoint_dict = {
                "epoch": completed_epoch,
                "optimizer": optimizer.state_dict(),
            }
            print("--------------------saving checkpoints--------------------")
            torch.save(
                checkpoint_dict,
                os.path.join(save_path, "best_states.pt"),
            )
            print("--------------------saving state_dict--------------------")
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_model.bin"),
            )
            best_clip_val_loss = cur_metrics["clip_val_loss"]

    eval_best(model, args, data, criterion, 20010321, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        choices=["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
        help="which dataset",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size"
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=100,
        help="epoch number"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=100,
        help="log every n steps"
    )
    # VIT-B-32
    # huggingface-cli download --resume-download laion/CLIP-ViT-B-32-laion2B-s34B-b79K --local-dir ./CLIP-ViT-B-32-laion2B-s34B-b79K
    # coca_ViT-L-14
    # huggingface-cli download --resume-download laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k --local-dir ./mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k
    # dataset
    # huggingface-cli download --resume-download osv5m/osv5m --local-dir ./osv5m  --repo-type dataset
    # huggingface-cli download --resume-download BAAI/bge-large-en-v1.5 --local-dir ./bge-large-en-v1.5
    # huggingface-cli download --resume-download nvidia/NV-Embed-v1 --local-dir ./NV-Embed-v1 --token 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
    # huggingface-cli download --resume-download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./all-MiniLM-L12-v2

    parser.add_argument(
        "--model", type=str, default="coca_ViT-L-14", help="model name"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/home/work/zhangruixing/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin",
        help="pretrained model after running main.py",
    )
    parser.add_argument(
        "--caption_loss_weight",
        type=float,
        default=1.0,
        help="weight on the autoregressive caption loss",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=132,
        help="random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="cpu or gpu"
    )
    parser.add_argument(
        "--name", type=str, default="test2", help="downstream name"
    )
    args = parser.parse_args()

    main(args)
