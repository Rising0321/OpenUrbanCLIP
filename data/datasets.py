from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm


class CoCaDataset(Dataset):
    def __init__(self, list_data=None, transform=None, tokenizer=None, city='Beijing', output_name=False):
        super().__init__()

        self.transform = transform
        self.tokenizer = tokenizer

        self.img_tensors = []
        self.captions = []
        self.caption_tokens = []
        self.city = city
        self.names = []

        self.output_name = output_name

        for item in tqdm(list_data, desc="Loading data"):
            try:
                im = Image.open(
                    item[0]
                ).convert("RGB")
                im = transform(im)
                self.img_tensors.append(im)
                self.caption_tokens.append(
                    self.tokenizer(item[1])
                )
                self.captions.append(item[1])
                self.names.append(item[0])
            except Exception as e:
                print("failed")
                print(item[0])
                continue

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        if self.output_name:
            return self.img_tensors[index], self.caption_tokens[index], self.names[index]
        else:
            return self.img_tensors[index], self.caption_tokens[index]


class LinearProbeDataSet(Dataset):
    def __init__(self, image_embeddings_, y_):
        super().__init__()
        self.img_embeds = image_embeddings_
        self.y = y_

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.img_embeds[index], np.float32(self.y[index])