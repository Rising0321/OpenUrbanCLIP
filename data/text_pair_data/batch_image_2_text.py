import os

os.system("CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 0")
os.system("CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 1")
os.system("CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 2")
os.system("CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 3")
os.system("CUDA_VISIBLE_DEVICES=1 python image2text.py --slice 4")
os.system("CUDA_VISIBLE_DEVICES=1 python image2text.py --slice 5")
os.system("CUDA_VISIBLE_DEVICES=1 python image2text.py --slice 7")

"CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 0 & CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 1 & CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 2 & CUDA_VISIBLE_DEVICES=0 python image2text.py --slice 3 & CUDA_VISIBLE_DEVICES=1 python image2text.py --slice 4 & CUDA_VISIBLE_DEVICES=1 python image2text.py --slice 5 & CUDA_VISIBLE_DEVICES=1 python image2text.py --slice 6"
