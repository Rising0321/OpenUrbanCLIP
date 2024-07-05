import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error)
from utils.utils import (
    set_random_seed,
)
from tqdm import tqdm, trange


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.project = nn.Linear(args.img_embedding_dim, 1)
        # self.project = nn.Linear(args.img_embedding_dim, args.project_dim)
        # self.activation = nn.ReLU() if args.activation == "relu" else nn.GELU()
        # self.dropout = nn.Dropout(args.drop_out)
        # self.predict = nn.Linear(args.project_dim, 1)

    def forward(self, image_latent):
        logits = self.project(image_latent)
        return logits.squeeze(1)
        # image_latent = self.project(image_latent)
        # image_latent = self.activation(image_latent)
        # image_latent = self.dropout(image_latent)
        # logits = self.predict(image_latent)
        # return logits.squeeze(1)


class MyDataSet(Dataset):
    def __init__(self, image_embeddings_, y_):
        super().__init__()
        self.img_embeds = image_embeddings_
        self.y = y_

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.img_embeds[index], np.float32(self.y[index])


def build_dataset(args, now_data, mean=False, std=False):
    flag = 0
    if mean == False:
        flag = 1
        temp = []
        for item in now_data:
            if item[4] > 0 and item[4] < 10000:
                temp.append(item[4])
        mean = np.mean(temp)
        std = np.std(temp)
    embed_dir = f"embeddings/{args.dataset}/{args.name}/"
    image_embeddings = []
    y = []
    for item in now_data:
        if item[4] < 0 or item[4] > 10000:
            continue
        coord = "%.4lf" % item[0], "%.4lf" % item[3], "%.4lf" % item[2], "%.4lf" % item[1]
        try:
            image_embedding = np.load(embed_dir + f"{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.npy")
            image_embeddings.append(image_embedding)
            y.append((item[4] - mean) / std)
        except:
            print(f"{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}.npy not found")
            continue
    if flag == 0:
        return DataLoader(MyDataSet(image_embeddings, y), batch_size=args.batch_size, shuffle=True)
    else:
        return DataLoader(MyDataSet(image_embeddings, y), batch_size=args.batch_size, shuffle=True), mean, std


def create_datasets(args):
    train_dataset_ratio = 0.7
    val_dataset_ratio = 0.15
    test_dataset_ratio = 0.2

    data = np.load(f"data/task_data/{args.dataset}/{args.indicator}.npy")
    np.random.shuffle(data)

    train_data = data[: int(len(data) * train_dataset_ratio)]
    val_data = data[int(len(data) * train_dataset_ratio): int(len(data) * (train_dataset_ratio + val_dataset_ratio))]
    test_data = data[int(len(data) * (train_dataset_ratio + val_dataset_ratio)):]

    train_dataset, mean, std = build_dataset(args, train_data)
    val_dataset = build_dataset(args, val_data, mean, std)
    test_dataset = build_dataset(args, test_data, mean, std)

    return train_dataset, val_dataset, test_dataset, mean, std


def calc(phase, epoch, all_predicts, all_y, loss):
    metrics = {}
    if loss is not None:
        metrics["loss"] = loss
    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)

    print(
        f"{phase} Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    return metrics


def train(model, criterion, optimizer, loader, args, epoch):
    device = torch.device(args.gpu)
    model.train()
    all_predictions = []
    all_truths = []
    total_loss = 0.0
    for embeds, y in loader:
        embeds = embeds.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()

        predicts = model(embeds)

        loss = criterion(predicts, y)
        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        all_predictions.extend(predicts.cpu().detach().numpy())
        all_truths.extend(y.cpu().detach().numpy())

    return calc("Train", epoch, all_predictions, all_truths, total_loss / len(loader))


def evaluate(model, loader, args, epoch):
    device = torch.device(args.gpu)
    model.eval()

    all_y, all_predicts = [], []
    with torch.no_grad():
        for embeds, y in loader:
            embeds = embeds.to(device=device)
            y = y.to(device=device)
            predicts = model(embeds)

            all_y.append(y.cpu().numpy())
            all_predicts.append(predicts.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_predicts = np.concatenate(all_predicts)
    return calc("Eval", epoch, all_predicts, all_y, None)


def main(args):
    device = args.gpu

    set_random_seed(args.seed)

    train_dataset, val_dataset, test_dataset, mean, std = create_datasets(args)

    model = MLP(args).to(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    best_val = float("inf")

    if not os.path.exists(f"./checkpoints/downstream/{args.dataset}"):
        os.makedirs(f"./checkpoints/downstream/{args.dataset}")

    checkpoints_dir = f"./checkpoints/downstream/{args.dataset}/{args.indicator}_best.pt"

    for epoch in range(args.epoch_num):
        train(model, criterion, optimizer, train_dataset, args, epoch)
        cur_metrics = evaluate(model, val_dataset, args, epoch)
        evaluate(model, test_dataset, args, "test")
        checkpoint_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if cur_metrics['mse'] < best_val:
            torch.save(
                checkpoint_dict,
                checkpoints_dir,
            )
            best_val = cur_metrics['mse']

    best_checkpoint = torch.load(
        checkpoints_dir, map_location=torch.device("cpu")
    )

    model.load_state_dict(best_checkpoint["state_dict"])
    model.to(device)
    evaluate(model, test_dataset, args, "test ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="Beijing",
        choices=["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
        help="which dataset",
    )
    parser.add_argument("--gpu", type=str, default="cuda:0")

    parser.add_argument(
        "--indicator",
        type=str,
        default="Carbon",
        choices=["Carbon", "Population", "Gdp"],
        help="indicator",
    )

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    parser.add_argument("--epoch_num", type=int, default=100000, help="epoch number")

    parser.add_argument(
        "--img_embedding_dim", type=int, default=768, help="image encoder output dim"
    )

    parser.add_argument(
        "--name", type=str, default="coca_ViT-L-14-test2", help="downstream name"
    )

    parser.add_argument("--seed", type=int, default=132, help="random seed")

    main(parser.parse_args())
