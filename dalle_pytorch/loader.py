from pathlib import Path
import torchvision
from random import randint, choice

import ipdb
st = ipdb.set_trace
import json
import PIL

from cub2011 import Cub2011

from torch.utils.data import Dataset
from torchvision import transforms as T


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
                 val=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle


        if folder == "cub200":
            self.dataset = Cub2011(root="/home/mprabhud/vision_datasets", download=True)
            self.dataset_name = "cub200"
        elif folder == "mnist":
            self.dataset = torchvision.datasets.MNIST(root="/home/mprabhud/vision_datasets", download=True, train=not val)
            self.dataset_name = "mnist"
            # st()
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
            # st()
            description = filename.split("/")[0].split(".")[1].replace("_"," ")
            image_tensor = self.image_transform(image)
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
        elif self.dataset_name == "mnist":
            image, target = self.dataset[ind]
            description = self.dataset.classes[target]
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
