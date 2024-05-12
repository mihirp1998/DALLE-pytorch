from pathlib import Path
import torchvision
from random import randint, choice
import numpy as np
import ipdb
st = ipdb.set_trace
import json
import PIL
import pickle
import torch
from cub2011 import Cub2011
from dalle_pytorch import mnist_corruptions
from torch.utils.data import Dataset
from torchvision import transforms as T

# class TokenizedDataset(Dataset):
#     """
#         Custom Dataset object to load and process data already tokenized and stored in
#         (X, y) pairs where X is input tokens and y is target tokens (either classes or masks)
#     """

#     def __init__(self, data):
#         self.data = data
#         # st()

#     def __len__(self):
#         return len(self.data[0])

#     def __getitem__(self, idx):
#         X = torch.tensor(self.data[0][idx], dtype=torch.long)
#         y = torch.tensor(self.data[1][idx], dtype=torch.long)
#         d = {}
#         d['src'] = X
#         d['trg'] = y

#         return d


class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 transparent=False,
                 tokenizer=None,
                 shuffle=False,
                 val=False,
                 pretokenized=False,
                 corruption=None,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        if corruption:
            assert corruption in mnist_corruptions.CORRUPTIONS
        self.corruption = corruption
        self.pretokenized = pretokenized

        if folder == "cub200":
            self.dataset = Cub2011(root="/home/mprabhud/vision_datasets", download=True)
            self.dataset_name = "cub200"
        elif folder == "mnist":
            self.dataset_name = "mnist"
            self.dataset = torchvision.datasets.MNIST(root="/home/mprabhud/vision_datasets", download=True, train=not val) # loading in both cases to get classes
            self.classes = self.dataset.classes
            if self.pretokenized:
                data_path = f'/home/mprabhud/sp/digen_data/{self.dataset_name}_vqgan.1024_total.pkl' # vqgan type hardcoded here, change if needed
                self.dataset = pickle.load(open(data_path, 'rb'))
        else:
            path = Path(folder)
            image_files = [
                *path.glob('**/*.png'), *path.glob('**/*.jpg'),
                *path.glob('**/*.jpeg'), *path.glob('**/*.bmp'),*path.glob('**/*.JPEG')
            ]
            imagenet_json = json.load(open("dalle_pytorch/imagenet.json")),
            imagenet_folder_name = {value[0]:value[1] for value in imagenet_json[0].values()}
            folder_names = [str(image_file).split("/")[-2] for image_file in image_files]
            class_names = [imagenet_folder_name[folder_name] for folder_name in folder_names]
            self.class_names= [class_name.replace("_"," ") for class_name in class_names]
            # self.class_names= [f"a photo of a {class_name}" for class_name in self.class_names]
            self.image_files = image_files
        # self.dataset = torchvision.datasets.CIFAR10(root="/home/mprabhud/vision_datasets", download=True)
        # text_files = [*path.glob('**/*.txt')]
        # text_files = {text_file.stem: text_file for text_file in text_files}
        # image_files = {image_file.stem: image_file for image_file in image_files}
        # keys = (image_files.keys() & text_files.keys())
        # self.keys = list(keys)
        # self.text_files = {k: v for k, v in text_files.items() if k in keys}
        # self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

        image_mode = 'RGBA' if transparent else 'RGB'

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert(image_mode)
            if img.mode != image_mode else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        if self.dataset_name == "cub200":
            return len(self.dataset)
        elif self.dataset_name == "mnist":
            if self.pretokenized:
                return len(self.dataset[0])
            return len(self.dataset)
        else:
            return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        if self.dataset_name == "cub200":
            image, target, filename = self.dataset[ind]
            description = filename.split("/")[0].split(".")[1].replace("_"," ")
            image_tensor = self.image_transform(image)
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
        elif self.dataset_name == "mnist":
            # st()
            if self.pretokenized:
                image, target = self.dataset[0][ind], self.dataset[1][ind]
                image_tensor = torch.tensor(image, dtype=torch.int32)
            else:
                image, target = self.dataset[ind]
            description = self.classes[target]
            # if self.corrupted_mnist:
            #     img_np = np.array(image)
            if not self.pretokenized:
                if self.corruption is not None:
                    # st()
                    np_img = np.array(image)
                    corruption_method = getattr(mnist_corruptions, self.corruption)
                    corrupted_img = corruption_method(np_img)
                    image = PIL.Image.fromarray(corrupted_img)
                    pass
                image_tensor = self.image_transform(image)
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
            # Image.fromarray((image_tensor * 255).to(torch.uint8).permute(1,2,0).numpy()).save("out.png")
            # st()
        else:
            description = self.class_names[ind]
            image_file = self.image_files[ind]

            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
            try:
                image_tensor = self.image_transform(PIL.Image.open(image_file))
            except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
                print(f"An exception occurred trying to load file {image_file}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor
