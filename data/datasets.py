from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm


class CoCaDataset(Dataset):
    def __init__(self, list_data=None, transform=None, tokenizer=None, city='Beijing'):
        super().__init__()

        self.transform = transform
        self.tokenizer = tokenizer

        self.img_paths = []
        self.img_tensors = []
        self.captions = []
        self.caption_tokens = []
        self.city = city
        for item in tqdm(list_data):
            try:
                im = Image.open(
                    item[0]
                ).convert("RGB")
                im = transform(im)
                self.img_tensors.append(im)
                self.caption_tokens.append(
                    self.tokenizer(item[1])
                )
                self.img_paths.append(item[0])
                self.captions.append(item[1])
            except Exception as e:
                print("failed")
                print(item[0])
                continue

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        print(self.img_tensors[index], self.caption_tokens[index])
        return self.img_tensors[index], self.caption_tokens[index]
