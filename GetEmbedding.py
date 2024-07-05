import os.path
import open_clip
import torch
import numpy as np
import argparse
from utils.utils import set_random_seed
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download laion/CLIP-ViT-B-32-laion2B-s34B-b79K --local-dir ./CLIP-ViT-B-32-laion2B-s34B-b79K
class ImageDataset(Dataset):
    def __init__(self, transform, image_path):
        super().__init__()
        self.transform = transform
        self.img_tensors = []
        self.name = []
        for root, _, files in os.walk(image_path):
            for file_name in tqdm(files):
                try:
                    if file_name.find("png") != -1:
                        _image_path = os.path.join(root, file_name)
                        _im = Image.open(_image_path).convert("RGB")
                        # im = transform(im).unsqueeze(0)  # [1, 3, 224, 224]
                        _im = transform(_im)  # [3, 224, 224]
                        self.img_tensors.append(_im)
                        name, _ = os.path.splitext(file_name)
                        self.name.append(name)
                except Exception as e:
                    print("failed")
                    print(file_name)
                    continue

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # print(self.x[index], self.y[index])
        return self.img_tensors[index], self.name[index]


def main(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model, pretrained=args.pretrained_model
    )
    model.to(device)
    save_path = f"./embeddings/{args.dataset}/{args.model}-{args.name}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    image_path = os.path.join("data/image_data", args.dataset)

    image_dataset = ImageDataset(transform, image_path)

    image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    for images, names in tqdm(image_dataloader):
        with torch.no_grad():
            images = images.to(device=device)
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
        "--model", type=str, default="coca_ViT-L-14", help="model name"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/home/work/zhangruixing/OpenUrbanCLIP/checkpoints/upstream/best_model.bin",
        help="pretrained model after running main.py",
    )
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=132, help="random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--name", type=str, default="test2", help="downstream name"
    )
    args = parser.parse_args()

    main(args)
