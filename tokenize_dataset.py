import gc
import os

__import__("builtins").st = __import__("ipdb").set_trace

import argparse
import sys

# import joblib as jb # waste of time
import pickle
import torch
from PIL import Image
from tqdm.auto import tqdm
import ipdb
st = ipdb.set_trace
from diffusion import datasets
# from vqgan import VQGAN_Engine, custom_to_pil
from dalle_pytorch.vae import VQGanVAE, VQGANDataset
from wilds import get_dataset

import ipdb
st = ipdb.set_trace

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="flowers", help="Dataset name")
    parser.add_argument("--vqgan", type=str, default="vq-f4", help="Dataset name")
    # parser.add_argument("--size", type=int, default=256, help="size")
    parser.add_argument("--save_dir", type=str, default="/home/mprabhud/sp/digen_data", help="Dataset name")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both", help="Dataset name")
    parser.add_argument("--resize_ratio", type=float, default=0.75, help="size")

    # partial args for processing in parallel
    parser.add_argument("--do_shared", action="store_true", help="Dataset name")
    parser.add_argument("--start", type=int, help="size", required='--do_shared' in sys.argv)
    parser.add_argument("--end", type=int, help="size", required='--do_shared' in sys.argv)
    parser.add_argument("--shard_num", type=int, help="size", required='--do_shared' in sys.argv)

    args, unknown = parser.parse_known_args()
    args.vqgan = 'vqgan.1024'
    print(f'Processing dataset {args.dataset}')

    if not args.split == 'test':
        print('Loading train dataset')
        train_dataset = datasets.get_target_dataset(args.dataset, train=True)
    if not args.split == 'train':
        print('Loading test dataset')
        test_dataset = datasets.get_target_dataset(args.dataset, train=False)


    if args.do_shared:
        print(f"Processing shard {args.shard_num} from {args.start} to {args.end} for {args.dataset}")
        print(f'Saving shared dataset to {args.save_dir}')
        if not args.split == 'test':
            # train_dataset = train_dataset[args.start:args.end]
            # train_dataset is torchvision.datasets.folder.ImageFolder - probably only for Imagenet, see datasets.py for type and how to splice

            # need to take subset of train_dataset
            train_dataset = torch.utils.data.Subset(train_dataset, range(args.start, args.end))
            print(f'Train dataset length: {len(train_dataset)}')

        if not args.split == 'train':
            test_dataset = test_dataset[args.start:args.end]
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.start, args.end))
            print(f'Test dataset length: {len(test_dataset)}')

    # st()
    total_dataset = train_dataset + test_dataset
    vqgan = VQGanVAE()
    IMAGE_SIZE = vqgan.image_size
    CHANNELS = vqgan.channels
    IMAGE_MODE = 'RGBA' if CHANNELS == 4 else 'RGB'
    RESIZE_RATIO = args.resize_ratio
    print("Codebook shape ", vqgan.codebook.shape)

    print(f'Saving dataset to {args.save_dir}')

    # if not args.split == 'test':
    vq_train_dataset = VQGANDataset(total_dataset, image_mode=IMAGE_MODE, image_size=IMAGE_SIZE, resize_ratio=RESIZE_RATIO)
    dataset_tokens, dataset_target = vqgan.encode_dataset(vq_train_dataset)

    if args.do_shared:
        pickle.dump([dataset_tokens, dataset_target], os.path.join(args.save_dir, f'{args.dataset}_{args.vqgan}_total_shard_{args.shard_num}.pkl'))
    else:
        pickle.dump([dataset_tokens, dataset_target], os.path.join(args.save_dir, f'{args.dataset}_{args.vqgan}_total.pkl'))

    del dataset_target, dataset_tokens
    gc.collect()

    # if not args.split == 'train':
    #     vq_test_dataset = VQGANDataset(test_dataset, image_mode=IMAGE_MODE, image_size=IMAGE_SIZE, resize_ratio=RESIZE_RATIO)
    #     dataset_tokens, dataset_target = vqgan.encode_dataset(vq_test_dataset)

    #     # for _, d in tqdm(enumerate(test_dataset), total=len(test_dataset)):
    #     #     img, target = d
    #     #     tokens = vqgan.encode(img)

    #     #     if torch.tensor(target).ndim > 1:
    #     #         target = vqgan.encode(target)

    #     #     dataset_tokens.append(tokens)
    #     #     dataset_target.append(target)
    #     if args.do_shared:
    #         jb.dump([dataset_tokens, dataset_target], os.path.join(args.save_dir, f'{args.dataset}_{args.vqgan}_test_shard_{args.shard_num}.pkl'))
    #     else:
    #         jb.dump([dataset_tokens, dataset_target], os.path.join(args.save_dir, f'{args.dataset}_{args.vqgan}_test.pkl'))
