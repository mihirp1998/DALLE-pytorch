import io
import sys
import os
import requests
import PIL
import warnings
import hashlib
import urllib
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt, log
from packaging import version

from omegaconf import OmegaConf
sys.path.insert(0,"/home/mprabhud/phd_projects/digen/taming-transformers/")
from taming.models.vqgan import VQModel, GumbelVQ
import importlib
import torchvision.transforms as T
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
import ipdb
st = ipdb.set_trace
from dalle_pytorch import distributed_utils

# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

# helpers methods

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))

def map_pixels(x, eps = 0.1):
    return (1 - 2 * eps) * x + eps

def unmap_pixels(x, eps = 0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

def download(url, filename = None, root = CACHE_PATH):
    if (
            not distributed_utils.is_distributed
            or distributed_utils.backend.is_local_root_worker()
    ):
        os.makedirs(root, exist_ok = True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if (
            distributed_utils.is_distributed
            and not distributed_utils.backend.is_local_root_worker()
            and not os.path.isfile(download_target)
    ):
        # If the file doesn't exist yet, wait until it's downloaded by the root worker.
        distributed_utils.backend.local_barrier()

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    if (
            distributed_utils.is_distributed
            and distributed_utils.backend.is_local_root_worker()
    ):
        distributed_utils.backend.local_barrier()
    return download_target

def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())

# package versions

def get_pkg_version(pkg_name):
    from pkg_resources import get_distribution
    return get_distribution(pkg_name).version

# pretrained Discrete VAE from OpenAI

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()
        assert version.parse(get_pkg_version('torch')) < version.parse('1.11.0'), 'torch version must be <= 1.10 in order to use OpenAI discrete vae'

        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))
        make_contiguous(self)

        self.channels = 3
        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim = 1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n)))

        z = F.one_hot(img_seq, num_classes = self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented

# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGANDataset:
    def __init__(self, ds, image_size, image_mode, resize_ratio):
        self.ds = ds
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert(image_mode)
            if img.mode != image_mode else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    # def preprocess(self, img, target_image_size=256):
    #     """
    #     input: RGB PIL image of shape (H, W, 3) - each value in [0, 255]
    #     output: torch tensor of shape (1, 3, target_image_size, target_image_size) - each value in [-1, 1]
    #     """
    #     s = min(img.size)

    #     if s < target_image_size:
    #         raise ValueError(f'min dim for image {s} < {target_image_size}')

    #     # already done in rescale
    #     r = target_image_size / s
    #     s = (round(r * img.size[1]), round(r * img.size[0]))
    #     img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    #     img = TF.center_crop(img, output_size=2 * [target_image_size])
    #     img = T.ToTensor()(img)
    #     img = preprocess_vqgan(img)
    #     return img

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        # print(img, target)
        # st()
        # # img = rescale(img) # same as preprocess
        # img = self.preprocess(img)

        # if isinstance(target, Image.Image):
        #     # target = rescale(target) # same
        #     target = self.preprocess(target)
        # img = T.ToTensor()(img)
        img = self.image_transform(img)

        return {'img': img, 'target': target}

    def __len__(self):
        return len(self.ds)


class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()

        if vqgan_model_path is None:
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'
            download(VQGAN_VAE_CONFIG_PATH, config_filename)
            download(VQGAN_VAE_PATH, model_filename)
            config_path = str(Path(CACHE_PATH) / config_filename)
            model_path = str(Path(CACHE_PATH) / model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])
        self.z_dim = config.model.params.ddconfig.z_channels

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model
        self.config = config
        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]

        self.num_layers = int(log(f)/log(2))
        self.channels = 3
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)
        self.codebook = self.model.quantize.embedding.weight.data
        print(f'VQGAN Codebook shape: {self.codebook.shape}')
        self.codebook_dict = {} # maps codebook vectors to their indices
        for i, v in enumerate(self.codebook):
            self.codebook_dict[tuple(v.cpu().numpy())] = i

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embed.weight if self.is_gumbel else self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented

    def encode_dataset(self, dataset: VQGANDataset):
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=256, #4,
                num_workers=1,
        )

        dataset_tokens = []
        dataset_target = []
        self.model = self.model.to('cuda')
        for d in tqdm(dataloader, total=len(dataloader)):
            img, target = d['img'], d['target']
            img = img.to('cuda')
            target = target.to('cuda')

            # st()

            # old code from digen
            # z, _, [_, _, indices] = self.model.encode(img)
            # # z1 = self.model.decode(z)
            # # custom_to_pil(z1[0]).save("/home/mprabhud/sp/x1.png")
            # z = z.detach()
            # z_processed = [x.squeeze(0).view(self.z_dim, -1).T for x in z]
            # z_prime = [self.replace_with_nearest(x) for x in z_processed]
            # z_prime_indices = [self.get_indices(x).cpu().numpy() for x in z_prime]

            z_prime_indices = self.get_codebook_indices(img) # (b, n)
            z_prime_indices = z_prime_indices.cpu().numpy().tolist()

            dataset_tokens.extend(z_prime_indices)

            if target.dim() > 3:
                # z, _, [_, _, indices] = self.model.encode(target)
                # z = z.detach()
                # z_processed = [x.squeeze(0).view(self.z_dim, -1).T for x in z]
                # z_prime = [self.replace_with_nearest(x) for x in z_processed]
                # z_prime_indices = [self.get_indices(x).cpu().numpy() for x in z_prime]

                z_prime_indices = self.get_codebook_indices(target) # (b, n)
                z_prime_indices = z_prime_indices.cpu().numpy().tolist()
                dataset_target.extend(z_prime_indices)
            else:
                target = target.cpu().numpy()
                dataset_target.extend([x for x in target])



        return dataset_tokens, dataset_target


