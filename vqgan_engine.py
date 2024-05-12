import sys

import ipdb

sys.path.insert(0, "./taming-transformers")


import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm


# def load_config(config_path, display=False):
#     config = OmegaConf.load(config_path)
#     if display:
#         print(yaml.dump(OmegaConf.to_container(config)))
#     return config

# def load_vqgan(config, ckpt_path=None, is_gumbel=False):
#     if is_gumbel:
#         model = GumbelVQ(**config.model.params)
#     else:
#         model = VQModel(**config.model.params)
#     if ckpt_path is not None:
#         sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
#         missing, unexpected = model.load_state_dict(sd, strict=False)
#     return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x




def rescale(img, target_image_size=256):
    s = min(img.size)
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    return img


class VQGAN_Engine:
    def __init__(self, vqgan):
        # config = load_config(config_path, display=False)
        # print(f'VQGAN Config: {config}')
        # self.z_dim = config['model']['params']['ddconfig']['z_channels']
        self.z_dim = vqgan.config.model.params.ddconfig.z_channels
        # self.model = load_vqgan(config, model_path).to(device)
        self.model = vqgan.cuda()
        self.device = 'cuda'
        self.codebook = self.model.quantize.embedding.weight.data
        print(f'VQGAN Codebook shape: {self.codebook.shape}')
        self.codebook_dict = {} # maps codebook vectors to their indices
        for i, v in enumerate(self.codebook):
            self.codebook_dict[tuple(v.cpu().numpy())] = i

    def preprocess(self, img, target_image_size=256):
        """
        input: RGB PIL image of shape (H, W, 3) - each value in [0, 255]
        output: torch tensor of shape (1, 3, target_image_size, target_image_size) - each value in [-1, 1]
        """

        # st()
        s = min(img.size)

        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')

        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        img = preprocess_vqgan(img)
        return img

    # def replace_with_nearest(self, z_p):
    #     """
    #     input: codebook tensor of shape (N, C), z_p tensor of shape (H*W, C)
    #     output: tensor of shape (H*W, C) with each row replaced by the nearest codebook vector
    #     """
    #     distances = torch.cdist(z_p, self.codebook) # (H*W, N)

    #     # find the index of the nearest codebook vector for each row
    #     nearest_indices = torch.argmin(distances, dim=1)

    #     # replace each row with the nearest codebook vector
    #     z_prime = torch.stack([self.codebook[i] for i in nearest_indices])
    #     return z_prime

    # def get_indices(self, z_p):
    #     """
    #     input: z_p tensor of shape (H*W, C), codebook_dict dict mapping codebook vectors to their indices
    #     output: tensor of shape (H*W) with each value replaced by the index of the nearest codebook vector
    #     """
    #     indices = []
    #     for i in range(z_p.shape[0]):
    #         indices.append(self.codebook_dict[tuple(z_p[i].cpu().numpy())])
    #     return torch.tensor(indices)


    def reconstruct(self, x):
        x = self.preprocess(x).to(self.device)
        z, _, [_, _, indices] = self.model.encode(x)
        xrec = self.model.decode(z)
        return z, xrec



    # def encode(self, img, size=256):
    #     img = rescale(img,target_image_size=size)
    #     # st()
    #     img = self.preprocess(img).to(self.device)
    #     z, _, [_, _, indices] = self.model.encode(img)
    #     z_processed = z.squeeze(0).view(self.z_dim, -1).T
    #     z_prime = self.replace_with_nearest(z_processed)
    #     z_prime_indices = self.get_indices(z_prime)
    #     return z_prime_indices.cpu().numpy()
    #     # xrec = model.decode(z)


    # def decode(self, vec):
    #     z_prime_index = vec
    #     z_prime = self.codebook[z_prime_index].T

    #     patches = z_prime.shape[1]
    #     patch_size = int(patches**0.5)

    #     z_prime = z_prime.view(1, -1, patch_size, patch_size).to(self.device)
    #     xrec_prime = self.model.decode(z_prime)
    #     return xrec_prime.squeeze(0).cpu() # remove batch dim if 1

    def decode_batch(self, batch_tokens):
        decoded_x = []
        for tokens in tqdm(batch_tokens, total=len(batch_tokens), desc='VQGAN decode'):
            decoded_x.append(custom_to_pil(self.decode(tokens)))
        return decoded_x


    def encode_dataset(self, dataset):

        dataset = VQGANDataset(dataset)
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=4, #4,
                num_workers=2,
        )

        dataset_tokens = []
        dataset_target = []

        for d in tqdm(dataloader, total=len(dataloader)):
            img, target = d['img'], d['target']
            img = img.to(self.device)
            target = target.to(self.device)


            # z, _, [_, _, indices] = self.model.encode(img)

            # # z1 = self.model.decode(z)
            # # custom_to_pil(z1[0]).save("/home/mprabhud/sp/x1.png")
            # # # sys.exit(0)

            # z = z.detach()
            # z_processed = [x.squeeze(0).view(self.z_dim, -1).T for x in z]
            # z_prime = [self.replace_with_nearest(x) for x in z_processed]
            # z_prime_indices = [self.get_indices(x).cpu().numpy() for x in z_prime]

            z_prime_indices = self.model.get_codebook_indices(img) # (b, n)
            z_prime_indices = z_prime_indices.cpu().numpy().tolist()

            dataset_tokens.extend(z_prime_indices)

            if target.dim() > 3:
                z, _, [_, _, indices] = self.model.encode(target)

                # z1 = self.model.decode(z)
                # custom_to_pil(z1[0]).save("/home/mprabhud/sp/t1.png")
                # sys.exit(0)


                z = z.detach()
                z_processed = [x.squeeze(0).view(self.z_dim, -1).T for x in z]
                z_prime = [self.replace_with_nearest(x) for x in z_processed]
                z_prime_indices = [self.get_indices(x).cpu().numpy() for x in z_prime]

                dataset_target.extend(z_prime_indices)
            else:
                target = target.cpu().numpy()
                dataset_target.extend([x for x in target])



        return dataset_tokens, dataset_target


class VQGANDataset:
    def __init__(self, ds):
        self.ds = ds

    def preprocess(self, img, target_image_size=256):
        """
        input: RGB PIL image of shape (H, W, 3) - each value in [0, 255]
        output: torch tensor of shape (1, 3, target_image_size, target_image_size) - each value in [-1, 1]
        """
        s = min(img.size)

        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')

        # already done in rescale
        # r = target_image_size / s
        # s = (round(r * img.size[1]), round(r * img.size[0]))
        # img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        # img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = T.ToTensor()(img)
        img = preprocess_vqgan(img)
        return img

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        img = rescale(img) # same as preprocess
        img = self.preprocess(img)

        if isinstance(target, Image.Image):
            target = rescale(target) # same
            target = self.preprocess(target)


        return {'img': img, 'target': target}

    def __len__(self):
        return len(self.ds)



if __name__ == '__main__':
    engine = VQGAN_Engine(
        "/home/mprabhud/phd_projects/digen/taming-transformers/logs/vqgan_imagenet_f16_16384/configs/model.yaml",
        "/home/mprabhud/phd_projects/digen/taming-transformers/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt",
        "cuda")

    # engine = VQGAN_Engine("/home/mprabhud/phd_projects/digen_sp/models/first_stage_models/vq-f4/config.yaml",
    #                       "/home/mprabhud/phd_projects/digen_sp/models/first_stage_models/vq-f4/model.ckpt",
    #                       "cuda")