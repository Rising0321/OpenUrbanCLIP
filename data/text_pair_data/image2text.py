import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser


def main():
    slice_max = 8
    parser = ArgumentParser()
    parser.add_argument("--slice", type=int, default=0)
    args = parser.parse_args()

    index = 3
    datasets = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"]
    dataset = datasets[index]

    print(dataset)

    image_dir = f"../image_data/{dataset}/"

    model = AutoModel.from_pretrained('/home/work/zhangruixing/MiniCPM-Llama3-V-2_5',
                                      torch_dtype=torch.float16,
                                      trust_remote_code=True)
    model = model.to(device='cuda')

    tokenizer = AutoTokenizer.from_pretrained('/home/work/zhangruixing/MiniCPM-Llama3-V-2_5',
                                              trust_remote_code=True)
    model.eval()

    res_array = []

    i = 0
    if not os.path.exists(f"{dataset}"):
        os.mkdir(f"{dataset}")

    dir_list = os.listdir(image_dir)
    len_dir = len(dir_list)

    if args.slice < slice_max - 1:
        dir_list = dir_list[args.slice * len_dir // slice_max: (args.slice + 1) * len_dir // slice_max]
    else:
        dir_list = dir_list[args.slice * len_dir // slice_max:]
    # print(dir_list)
    for file in tqdm(dir_list):
        path = os.path.join(image_dir, file)
        image = Image.open(path).convert('RGB')
        question = 'Offer a comprehensive summary of human activity, urban infrastructure, andenvironments in aerial image'
        msgs = [{'role': 'user', 'content': question}]

        ## if you want to use streaming, please make sure sampling=True and stream=True
        ## the model.chat will return a generator
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            stream=True
        )

        generated_text = ""
        for new_text in res:
            generated_text += new_text

        res_array.append([file, generated_text])

    np.save(f"{dataset}/image2text-{args.slice}.npy", res_array, allow_pickle=True)


main()
