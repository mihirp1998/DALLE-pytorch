import argparse
import sys
sys.path.insert(0,"/home/mprabhud/phd_projects/digen/taming-transformers/")
import ipdb
st = ipdb.set_trace
from pathlib import Path
import time
from glob import glob
import hydra
import os
import shutil
from omegaconf import DictConfig
import torch
import wandb  # Quit early if user doesn't have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dalle_pytorch import __version__
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import TextImageDataset
# libraries needed for webdataset support
import webdataset as wds
from torchvision import transforms as T
from PIL import Image
from io import BytesIO

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    # constants
    # st()
    from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer
    
    
    
    WEBDATASET_IMAGE_TEXT_COLUMNS = tuple(args.wds.split(','))
    ENABLE_WEBDATASET = True if len(WEBDATASET_IMAGE_TEXT_COLUMNS) == 2 else False

    DALLE_OUTPUT_FILE_NAME = args.dalle_output_file_name + ".pt"

    VAE_PATH = args.vae_path
    VQGAN_MODEL_PATH = args.vqgan_model_path
    VQGAN_CONFIG_PATH = args.vqgan_config_path
    DALLE_PATH = args.dalle_path
    RESUME = exists(DALLE_PATH)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    LEARNING_RATE = args.learning_rate
    GRAD_CLIP_NORM = args.clip_grad_norm
    LR_DECAY = args.lr_decay
    SAVE_EVERY_N_STEPS = args.save_every_n_steps
    KEEP_N_CHECKPOINTS = args.keep_n_checkpoints

    MODEL_DIM = args.dim
    TEXT_SEQ_LEN = args.text_seq_len
    DEPTH = args.depth
    HEADS = args.heads
    DIM_HEAD = args.dim_head
    REVERSIBLE = args.reversible
    LOSS_IMG_WEIGHT = args.loss_img_weight
    FF_DROPOUT = args.ff_dropout
    ATTN_DROPOUT = args.attn_dropout
    STABLE = args.stable_softmax
    SHIFT_TOKENS = args.shift_tokens
    ROTARY_EMB = args.rotary_emb

    ATTN_TYPES = tuple(args.attn_types.split(','))
    SHARED_ATTN_IDS = tuple(args.shared_attn_ids.split(',')) if exists(args.shared_attn_ids) else None
    SHARED_FF_IDS = tuple(args.shared_ff_ids.split(',')) if exists(args.shared_ff_ids) else None
    SHARE_INPUT_OUTPUT_EMB = args.share_input_output_emb

    DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'
    # st()

    if not ENABLE_WEBDATASET:
        # quit early if you used the wrong folder name
        pass
        # assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'
    else:
        # quit early if no tar files were found
        if Path(args.image_text_folder).is_dir():
            DATASET = [str(p) for p in Path(args.image_text_folder).glob("**/*") if ".tar" in str(p).lower()] # .name
            assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(args.image_text_folder)
            print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), args.image_text_folder))
        elif ('http://' in args.image_text_folder.lower()) | ('https://' in args.image_text_folder.lower()):
            DATASET = f"pipe:curl -L -s {args.image_text_folder} || true"
            print('Found {} http(s) link under given path!'.format(len(DATASET), args.image_text_folder))
        elif 'gs://' in args.image_text_folder.lower():
            DATASET = f"pipe:gsutil cat {args.image_text_folder} || true"
            print('Found {} GCS link under given path!'.format(len(DATASET), args.image_text_folder))
        elif '.tar' in args.image_text_folder:
            DATASET = args.image_text_folder
            print('Found WebDataset .tar(.gz) file under given path {}!'.format(args.image_text_folder))
        else:
            raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(args.image_text_folder))

    # initialize distributed backend
    # st()
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    
    using_deepspeed = \
        distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

    is_root = distr_backend.is_root_worker()

    # tokenizer

    if exists(args.bpe_path):
        klass = HugTokenizer if args.hug else YttmTokenizer
        tokenizer = klass(args.bpe_path)
    elif args.chinese:
        tokenizer = ChineseTokenizer()

    # reconstitute vae

    if RESUME:
        dalle_path = Path(DALLE_PATH)
        if using_deepspeed:
            cp_dir = cp_path_to_dir(dalle_path, 'ds')
            assert cp_dir.is_dir(), \
                f'DeepSpeed checkpoint directory {cp_dir} not found'
            dalle_path = cp_dir / DEEPSPEED_CP_AUX_FILENAME
        else:
            assert dalle_path.exists(), 'DALL-E model file does not exist'
        loaded_obj = torch.load(str(dalle_path), map_location='cpu')

        dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
        opt_state = loaded_obj.get('opt_state')
        scheduler_state = loaded_obj.get('scheduler_state')
        # st()
        if vae_params is not None:
            vae = DiscreteVAE(**vae_params)
        elif args.taming:
            vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae = OpenAIDiscreteVAE()

        resume_epoch = loaded_obj.get('epoch', 0)
    else:
        if exists(VAE_PATH):
            vae_path = Path(VAE_PATH)
            assert vae_path.exists(), 'VAE model file does not exist'
            assert not vae_path.is_dir(), \
                ('Cannot load VAE model from directory; please use a '
                'standard *.pt checkpoint. '
                'Currently, merging a DeepSpeed-partitioned VAE into a DALLE '
                'model is not supported.')

            loaded_obj = torch.load(str(vae_path))

            vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

            vae = DiscreteVAE(**vae_params)
            vae.load_state_dict(weights)
        else:
            if is_root:
                print('using pretrained VAE for encoding images to tokens')
            vae_params = None

            if args.taming:
                vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
            else:
                vae = OpenAIDiscreteVAE()

        dalle_params = dict(
            num_text_tokens=tokenizer.vocab_size,
            text_seq_len=TEXT_SEQ_LEN,
            dim=MODEL_DIM,
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD,
            reversible=REVERSIBLE,
            loss_img_weight=LOSS_IMG_WEIGHT,
            attn_types=ATTN_TYPES,
            ff_dropout=FF_DROPOUT,
            attn_dropout=ATTN_DROPOUT,
            stable=STABLE,
            shift_tokens=SHIFT_TOKENS,
            rotary_emb=ROTARY_EMB,
            shared_attn_ids=SHARED_ATTN_IDS,
            shared_ff_ids=SHARED_FF_IDS,
            share_input_output_emb=SHARE_INPUT_OUTPUT_EMB,
        )
        resume_epoch = 0

    IMAGE_SIZE = vae.image_size
    CHANNELS = vae.channels
    TRANSPARENT = CHANNELS == 4
    IMAGE_MODE = 'RGBA' if CHANNELS == 4 else 'RGB'

    # configure OpenAI VAE for float16s

    if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
        vae.enc.blocks.output.conv.use_float16 = True

    # helpers

    def group_weight(model):
        group_decay, group_no_decay = [], []
        for params in model.named_parameters():
            if 'transformer' in params[0]:
                if 'bias' in params[0] or 'norm' in params[0]:
                    group_no_decay.append(params[1])
                    continue
            group_decay.append(params[1])

        assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups


    # create dataset and dataloader

    is_shuffle = not distributed_utils.using_backend(distributed_utils.HorovodBackend)

    imagepreproc = T.Compose([
        T.Lambda(lambda img: img.convert(IMAGE_MODE)
        if img.mode != IMAGE_MODE else img),
        T.RandomResizedCrop(IMAGE_SIZE,
                            scale=(args.resize_ratio, 1.),
                            ratio=(1., 1.)),
        T.ToTensor(),
    ])

    def imagetransform(b):
        return Image.open(BytesIO(b))

    def tokenize(s):
        return tokenizer.tokenize(
            s.decode('utf-8'),
            TEXT_SEQ_LEN,
            truncate_text=args.truncate_captions).squeeze(0)

    if ENABLE_WEBDATASET:
        DATASET_SIZE = int(1e9) # You need to set a nominal length for the Dataset in order to avoid warnings from DataLoader

        myimg, mycap = WEBDATASET_IMAGE_TEXT_COLUMNS
        image_text_mapping = {
            myimg: imagetransform,
            mycap: tokenize
        }
        image_mapping = {
            myimg: imagepreproc
        }

        def filter_dataset(item): # For e.g. C@H which (rarely) has no caption available.
            if mycap not in item:
                return False
            if myimg not in item:
                return False
            return True

        w_dataset = wds.WebDataset(DATASET, handler=wds.warn_and_continue)
        filtered_dataset = w_dataset.select(filter_dataset)
        ds = filtered_dataset.map_dict(**image_text_mapping).map_dict(**image_mapping).to_tuple(mycap, myimg).batched(BATCH_SIZE / distr_backend.get_world_size(), partial=True)
    else:
        ds = TextImageDataset(
            args.image_text_folder,
            text_len=TEXT_SEQ_LEN,
            image_size=IMAGE_SIZE,
            transparent=TRANSPARENT,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=is_shuffle,
        )
        assert len(ds) > 0, 'dataset is empty'

    if is_root:
        if not ENABLE_WEBDATASET:
            print(f'{len(ds)} image-text pairs found for training')

    # data sampler

    data_sampler = None

    if not is_shuffle:
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=distr_backend.get_world_size(),
            rank=distr_backend.get_rank()
        )

    # WebLoader for WebDataset and DeepSpeed compatibility

    if ENABLE_WEBDATASET:
        dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=4) # optionally add num_workers=2 (n) argument
        number_of_batches = DATASET_SIZE // (BATCH_SIZE * distr_backend.get_world_size())
        dl = dl.slice(number_of_batches)
        dl.length = number_of_batches
    else:
        # Regular DataLoader for image-text-folder datasets
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)

    # initialize DALL-E

    dalle = DALLE(vae=vae, args=args, **dalle_params)
    num_params = sum([param.numel() for param in dalle.parameters() if param.requires_grad])
    args.num_params = num_params
    # st()

    if not using_deepspeed:
        if args.fp16:
            dalle = dalle.half()
        dalle = dalle.cuda()

    if RESUME and not using_deepspeed:
        dalle.load_state_dict(weights)

    # optimizer

    opt = Adam(get_trainable_params(dalle), lr=LEARNING_RATE)

    if RESUME and opt_state:
        opt.load_state_dict(opt_state)

    # scheduler

    scheduler = None

    if LR_DECAY:
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        if RESUME and scheduler_state:
            scheduler.load_state_dict(scheduler_state)

    # experiment tracker

    if is_root:
        # st()
        model_config = dict(
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD
        )

        run = wandb.init(
            project=args.wandb_name,
            entity=args.wandb_entity,
            resume=False,
            config=dict(args),
            mode="disabled" if args.debug else "online"
        )

    # distribute

    distr_backend.check_batch_size(BATCH_SIZE)
    deepspeed_config = {
        'train_batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': args.ga_steps,
        'gradient_clipping': GRAD_CLIP_NORM,
        'fp16': {
            'enabled': args.fp16,
        },
        'amp': {
            'enabled': args.amp,
            'opt_level': 'O1',
        },
        "flops_profiler": {
            "enabled": args.flops_profiler,
            "profile_step": 200,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None # TODO Can't get this to work.
        },
    }

    if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2:
        print(f"Checkpoints made with DeepSpeed ZeRO Stages 2 and 3 will be stored in deepspeed checkpoint folder")
        print(f"As such, they will require DeepSpeed as a dependency in order to resume from or generate with.")
        print("See the deespeed conversion script for details on how to convert your ZeRO stage 2/3 checkpoint to a single file.")
        print("If using a single GPU, consider running with apex automatic mixed precision instead for a similar speedup to ZeRO.")
        time.sleep(2)

    (distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
        args=args,
        model=dalle,
        optimizer=opt,
        model_parameters=get_trainable_params(dalle),
        training_data=(
            (None if ENABLE_WEBDATASET else ds)
            if using_deepspeed
            else dl
        ),
        # Do not pass the LR scheduler to DeepSpeed so we can manually
        # advance it.
        lr_scheduler=scheduler if LR_DECAY and not using_deepspeed else None,
        config_params=deepspeed_config,
    )
    # Prefer scheduler in `deepspeed_config`.

    if LR_DECAY and distr_scheduler is None:
        distr_scheduler = scheduler

    avoid_model_calls = using_deepspeed and args.fp16

    if RESUME and using_deepspeed:
        distr_dalle.load_checkpoint(str(cp_dir))


    def save_model(path, epoch=0):
        save_obj = {
            'hparams': dalle_params,
            'vae_params': vae_params,
            'epoch': epoch,
            'version': __version__,
            'vae_class_name': vae.__class__.__name__
        }

        if using_deepspeed:
            cp_dir = cp_path_to_dir(path, 'ds')

            if KEEP_N_CHECKPOINTS is not None and is_root:
                checkpoints = sorted(glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
                for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
                    shutil.rmtree(checkpoint)

            distr_dalle.save_checkpoint(cp_dir, client_state=save_obj)

            if not is_root:
                return

            # Save auxiliary values so we can reuse the standard routine
            # for loading.
            save_obj = {
                **save_obj,
                # Save a nonsense value that directs the user to
                # further help.
                'weights': (
                    'To get a working standard checkpoint, '
                    'look into consolidating DeepSpeed checkpoints.'
                ),
            }
            torch.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
            if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2: # see https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints
                return

        if not is_root:
            return

        save_obj = {
            **save_obj,
            'weights': dalle.state_dict(),
            'opt_state': opt.state_dict(),
            'scheduler_state': (scheduler.state_dict() if scheduler else None)
        }

        torch.save(save_obj, path)

    def save_artifact(model_config, model_path, name = 'trained-dalle'):
        model_artifact = wandb.Artifact(name, type='model', metadata=dict(model_config))
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

    # training

    # Saves a checkpoint before training begins to fail early when mis-configured.
    # See https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints

    save_model(DALLE_OUTPUT_FILE_NAME, epoch=resume_epoch)
    global_steps = 0

    for epoch in range(resume_epoch, EPOCHS):
        if data_sampler:
            data_sampler.set_epoch(epoch)

        for i, (text, images) in enumerate((dl if ENABLE_WEBDATASET else distr_dl)):
            # st()
            global_steps += 1
            if i % 10 == 0 and is_root:
                t = time.time()

            if args.fp16:
                images = images.half()

            text, images = map(lambda t: t.cuda(), (text, images))
            
            loss,_ = distr_dalle(text, images, return_loss=True)
            # st()
            forward_loss = loss.clone()
            
            if args.mode == "forward_forward":
                inverse_loss,accuracy = distr_dalle(text, images, return_loss=True, inverse_mapping=True)    
                loss += inverse_loss
            elif args.mode == "forward_reverse_partial":
                inverse_loss,accuracy = distr_dalle(text, images, return_loss=True, inverse_mapping=True, reverse_model=True)
                loss += inverse_loss

            if using_deepspeed:
                distr_dalle.backward(loss)
                distr_dalle.step()
                # Gradients are automatically zeroed after the step
            else:
                loss.backward()
                clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
                distr_opt.step()
                distr_opt.zero_grad()

            # Collective loss, averaged
            avg_loss = distr_backend.average_all(loss)
            avg_accuracy = distr_backend.average_all(accuracy)
            avg_forward_loss = distr_backend.average_all(forward_loss)
            # st()
            
            if args.mode == "forward_forward" or args.mode == "forward_reverse_partial":
                avg_inverse_loss = distr_backend.average_all(inverse_loss)
            else:
                avg_inverse_loss =  0.0

            log = {}
            # st()
            if i % 10 == 0 and is_root:
                print(epoch, i, f'loss - {avg_loss.item()}')

                log = {
                    **log,
                    'epoch': epoch,
                    'iter': i,
                    "accuracy": avg_accuracy,
                    'loss': avg_loss.item(),
                    'forward_loss': avg_forward_loss.item(),
                    'inverse_loss': avg_inverse_loss.item()
                }
                # st()

            if i % SAVE_EVERY_N_STEPS == 0:
                # st()
                run_name = wandb.run.name 
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                save_model(f"checkpoints/{run_name}/model.pt", epoch=epoch)
            # st()
            # print(i)
            if global_steps % args.log_images_freq == 0 and is_root:
                # print("logging image")
                # st()
                sample_text = text[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)

                if not avoid_model_calls:
                    # CUDA index errors when we don't guard this
                    image = dalle.generate_images(text[:1], filter_thres=0.9)  # topk sampling at 0.9

                if not avoid_model_calls:
                    log['image'] = wandb.Image(image, caption=decoded_text)

            if i % 10 == 9 and is_root:
                sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
                log["sample_per_sec"] = sample_per_sec
                print(epoch, i, f'sample_per_sec - {sample_per_sec}')

            if i == 201 and args.flops_profiler:
                raise StopIteration("Profiler has finished running. Stopping training early.")

            if is_root:
                wandb.log(log)

        if LR_DECAY:
            distr_scheduler.step(avg_loss)

        save_model(f"checkpoints/{run_name}/model.pt", epoch=epoch)

        # if is_root:
        #     # save trained model to wandb as an artifact every epoch's end
        #     save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)

    save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

    if is_root:
        # wandb.save(DALLE_OUTPUT_FILE_NAME)
        # save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)
        wandb.finish()



if __name__ == "__main__":
    main()
