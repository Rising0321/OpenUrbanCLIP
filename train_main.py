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
    read_files,
)
from tqdm import tqdm
from loguru import logger


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, criterion, data, epoch, optimizer, args, logger):
    """To train one epoch."""
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model.train()
    dataloader = data["train_loader"]
    num_batches_per_epoch = len(dataloader)
    sample_digits = math.ceil(math.log(len(dataloader) * args.batch_size + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()  # data loading time
    end = time.time()

    batch_size = args.batch_size
    total_batches = len(dataloader.dataset) // batch_size
    if len(dataloader.dataset) % batch_size != 0:
        total_batches += 1

    for batch_count, batch in tqdm(enumerate(dataloader), desc="Training: ", total=total_batches):
        step = num_batches_per_epoch * epoch + batch_count
        (
            images,
            texts,
        ) = batch  # images: [batch_size, 3, 224, 224], texts: [batch_size, 77]
        images = images.to(device=device)
        texts = texts.to(device=device)

        print(texts)

        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        if texts.ndim == 3:  # torch.Size([batch_size, 1, 77])
            texts = texts.squeeze(1)

        # print("images.shape: {}".format(images.shape))
        # print("texts.shape: {}".format(texts.shape))
        model_out = model(images, texts)
        """
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp()
        }
        """
        logit_scale = model_out["logit_scale"]

        losses = criterion(**model_out, output_dict=True)
        # {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        total_loss = sum(losses.values())
        losses["loss"] = total_loss

        # backward(total_loss, scaler)
        total_loss.backward()

        # if args.grad_clip_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count += 1
        if step % args.log_every_n_steps == 0:
            batch_size = len(images)
            num_samples = step * batch_size
            samples_per_epoch = (
                    num_batches_per_epoch * batch_size
            )  # sample size per epoch
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = batch_size / batch_time_m.val
            logger.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                # f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "scale": logit_scale_scalar,
                # "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                logger.info({name: val, "step": step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, logger):
    metrics = {}
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["val_loader"]
    num_samples = 0
    samples_per_val = len(dataloader) * args.batch_size  # sample size per epoch

    cumulative_loss = 0.0
    cumulative_gen_loss = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        batch_size = args.batch_size
        total_batches = len(dataloader.dataset) // batch_size
        if len(dataloader.dataset) % batch_size != 0:
            total_batches += 1

        for i, batch in tqdm(enumerate(dataloader), desc="Validation: ", total=total_batches):
            images, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            if texts.ndim == 3:
                texts = texts.squeeze(1)
            model_out = model(images, texts)
            image_features = model_out["image_features"]
            text_features = model_out["text_features"]
            logit_scale = model_out["logit_scale"]
            # print(image_features.shape)
            # print(text_features.shape)
            # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            # however, system RAM is easily exceeded and compute time becomes problematic
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=device).long()
            total_loss = (  # contrastive loss
                                 F.cross_entropy(logits_per_image, labels)
                                 + F.cross_entropy(logits_per_text, labels)
                         ) / 2

            gen_loss = maybe_compute_generative_loss(model_out)

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size

            if i % 100 == 0:
                logger.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                )

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    logger.info(
                        f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"
                    )

        val_metrics = get_clip_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        loss = cumulative_loss / num_samples
        metrics.update(
            {
                **val_metrics,
                "clip_val_loss": loss.item(),
                "epoch": epoch,
                "num_samples": num_samples,
            }
        )
        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({"val_generative_loss": gen_loss.item()})

    logger.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    for name, val in metrics.items():
        logger.info({f"val/{name}": val, "epoch": epoch})

    return metrics


def inference(model, data, args, logger):
    """test on test dataset."""
    metrics = {}
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = data["test_loader"]
    num_samples = 0
    samples_per_val = len(dataloader) * args.batch_size  # sample size per epoch

    cumulative_loss = 0.0
    cumulative_gen_loss = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        batch_size = args.batch_size
        total_batches = len(dataloader.dataset) // batch_size
        if len(dataloader.dataset) % batch_size != 0:
            total_batches += 1

        for i, batch in tqdm(enumerate(dataloader), desc="Testing: ", total=total_batches):
            images, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            if texts.ndim == 3:
                texts = texts.squeeze(1)
            model_out = model(images, texts)
            image_features = model_out["image_features"]
            text_features = model_out["text_features"]
            logit_scale = model_out["logit_scale"]

            # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            # however, system RAM is easily exceeded and compute time becomes problematic
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=device).long()
            total_loss = (  # contrastive loss
                                 F.cross_entropy(logits_per_image, labels)
                                 + F.cross_entropy(logits_per_text, labels)
                         ) / 2

            gen_loss = maybe_compute_generative_loss(model_out)

            cumulative_loss += total_loss * batch_size
            num_samples += batch_size

            if i % 100 == 0:
                logger.info(
                    f"Test : [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                )

                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                    logger.info(
                        f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"
                    )

        val_metrics = get_clip_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        loss = cumulative_loss / num_samples
        metrics.update(
            {**val_metrics, "clip_test_loss": loss.item(), "num_samples": num_samples}
        )
        if gen_loss is not None:
            gen_loss = cumulative_gen_loss / num_samples
            metrics.update({"test_generative_loss": gen_loss.item()})

    logger.info(
        f"Test: " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    for name, val in metrics.items():
        logger.info({f"test/{name}": val})

    return metrics


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


def eval_best(model, args, data):
    best_checkpoint = torch.load(
        os.path.join("./checkpoints/upstream", "best_model.bin"),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(best_checkpoint)
    model.to(args.device)
    test_metric = inference(model, data, args, logger)
    logger.info("test metric: {}".format(test_metric))


def init_model(args):
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="ViT-B-32", pretrained=args.pretrained_model
    )

    model.to(args.device)

    logger.info("model parameters: {}".format(count_trainable_parameters(model)))

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )

    criterion = open_clip.CoCaLoss(
        caption_loss_weight=args.caption_loss_weight,
        clip_loss_weight=1,
    )

    return model, transform, tokenizer, optimizer, criterion


def main(args):
    init(args)

    model, transform, tokenizer, optimizer, criterion = init_model(args)

    # create datasets
    data = create_datasets(
        args, transform, tokenizer
    )

    best_clip_val_loss = float("inf")

    for epoch in range(args.epoch_num):
        logger.info("Start epoch {}".format(epoch))
        print("Start epoch {}".format(epoch))
        train_one_epoch(model, criterion, data, epoch, optimizer, args, logger)
        completed_epoch = epoch + 1

        cur_metrics = evaluate(model, data, completed_epoch, args, logger)

        if cur_metrics["clip_val_loss"] < best_clip_val_loss:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                checkpoint_dict,
                os.path.join("./checkpoints/upstream", "best_states.pt"),
            )
            torch.save(
                model.state_dict(),
                os.path.join("./checkpoints/upstream", "best_model.bin"),
            )
            best_clip_val_loss = cur_metrics["clip_val_loss"]
    eval_best(model, args, data)


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
        default=8,
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
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/home/work/zhangruixing/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
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
    args = parser.parse_args()

    main(args)
