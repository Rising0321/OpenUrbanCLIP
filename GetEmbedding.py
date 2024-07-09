import os.path
import open_clip
import torch
import numpy as np
import argparse

from data.datasets import CoCaDataset
from utils.utils import set_random_seed, load_data, init_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


def main(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform, tokenizer, optimizer, criterion = init_model(args)

    save_path = f"./embeddings/{args.dataset}/{args.model}-{args.name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data = load_data(args)

    dataset = CoCaDataset(data, transform, tokenizer, args.dataset, output_name=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model.eval()

    for batch in tqdm(dataloader):
        with torch.no_grad():
            images, texts, names = batch
            images = images.to(device=device)
            texts = texts.to(device=device)

            embeds = model.encode_image(images)

            # print(embeds.shape) # [batch_size,768]
            for embed, name in zip(embeds, names):
                file_name = f'{name}.npy'
                path = os.path.join(save_path, file_name)
                np.save(path, embed.detach().cpu().numpy())


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
        "--model", type=str, default="coca_ViT-L-14", help="model name",
        choices=["coca_ViT-L-14", "coca_ViT-B-32"]
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/home/work/zhangruixing/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin",
        help="pretrained model after running main.py",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=132, help="random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--name", type=str, default="test2", help="downstream name"
    )
    # ------------------------------------------------------------
    parser.add_argument(
        "--lr", type=float, default=12345, help="not used but necessary like me"
    )
    parser.add_argument(
        "--caption_loss_weight", type=float, default=12345, help="not used but necessary like me"
    )
    args = parser.parse_args()

    main(args)
