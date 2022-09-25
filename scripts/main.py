import os
dataset_root = "Data/flowerImage"
kernel_root = os.path.join("/", "kaggle", "input", "text-to-image-xlnet-pytorch")

import io
import h5py
import torch
import ipywidgets
import numpy as np
from torch import nn
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
# import genrator from genrator.py
import Generator 
import textencoder
import Discrimintor


class TrainDataset:
    def __init__(self, dataset_root, kernel_root):
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

        if os.path.exists(os.path.join(kernel_root, "data.npy")):
            self.data = np.load(os.path.join(kernel_root, "data.npy"), allow_pickle=True)
        else:
            f = h5py.File(os.path.join(dataset_root, "data", "flowers", "flowers.hdf5"), mode="r")
            self.data = self.prepareData(f['train'])
        np.save('data.npy', self.data)
        self.max_seq_len = max(map(lambda x: len(x["text"]["input_ids"]), self.data))

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.RandomRotation(degrees=(270, 270)),
            transforms.RandomVerticalFlip(p=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])


    def prepareData(self, data):
        preparedData = []
        for idx, img_name in enumerate(tqdm(data)):
            image = np.array(Image.open(io.BytesIO(bytes(np.array(data[img_name]['img'])))).resize((256,256)))
            text = np.array(data[img_name]['txt']).item().strip()
            input_ids = self.tokenizer.encode(str(text), add_special_tokens=True, max_length=512, truncation=True)
            token_type_ids = [0] * (len(input_ids) - 1) + [1]
            attention_mask = [1] * len(token_type_ids)
            preparedData.append({
                "image": image,
                "text": {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask
                },
            })
        return preparedData

    def padTokens(self, text_dict):
        pad_len = self.max_seq_len - sum(text_dict["attention_mask"])
        text_dict['input_ids'] =  [5] * pad_len + text_dict['input_ids'] # <pad> = 5
        text_dict['token_type_ids'] =  [2] * pad_len + text_dict['token_type_ids']
        text_dict['attention_mask'] = [0] * pad_len + text_dict['attention_mask']   
        return text_dict

    @staticmethod
    def collate_fn_module(batch, idx):
        images, texts = [], {}
        for data in batch:
            images.append(data[idx][0])
            for key in data[idx][1]:
                if key not in texts:
                    texts[key] = []
                texts[key].append(data[0][1][key])

        images = torch.stack(images).to(device)
        for key in texts:
            texts[key] = torch.tensor(texts[key]).to(device)
        return images, texts

    def collate_fn(self, batch):
        right_images, right_texts = self.collate_fn_module(batch, 0)
        wrong_images, wrong_texts = self.collate_fn_module(batch, 1)
        return (right_images, right_texts), (wrong_images, wrong_texts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, right_idx):
        right_data = self.data[right_idx].copy()
        right_image = self.transforms(Image.fromarray(right_data["image"]))
        right_text = self.padTokens(right_data["text"].copy())

        wrong_idx = np.random.choice([(i) for i in range(len(self.data)) if i != right_idx])
        wrong_data = self.data[wrong_idx].copy()
        wrong_image = self.transforms(Image.fromarray(wrong_data["image"]))
        wrong_text = self.padTokens(wrong_data["text"].copy())
        return (right_image, right_text), (wrong_image, wrong_text)

#%%time
train_dataset = TrainDataset(dataset_root, kernel_root)