# Modified from Meta DiT: https://github.com/facebookresearch/DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from glob import glob

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from diffusers.models import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from opendit.diffusion import create_diffusion
from opendit.models.dit import DiT_models
from opendit.models.latte import Latte_models
from opendit.models.mmdit import MMDiT_models
from opendit.models.mmdit_latte import MMLatte_models
from opendit.utils.ckpt_utils import create_logger, load, record_model_param_shape, save
from opendit.utils.data_utils import prepare_dataloader
from opendit.utils.operation import model_sharding
from opendit.utils.pg_utils import ProcessGroupManager
from opendit.utils.train_utils import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, update_ema
from opendit.utils.video_utils import DatasetFromCSV, get_transforms_image, get_transforms_video
from opendit.vae.wrapper import AutoencoderKLWrapper, MultiModalVAEWrapper

# SHRIMP
from opendit.utils.DatasetBuilder import DatasetBuilder
from opendit.utils.dataset import SatelliteDataset

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({}, seed=args.global_seed)
    coordinator = DistCoordinator()
    device = get_current_device()

    # ==============================
    # Setup an experiment folder
    # ==============================
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(args.outputs, exist_ok=True)
    experiment_index = len(glob(f"{args.outputs}/*"))
    # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    model_string_name = args.model.replace("/", "-")
    # Create an experiment folder
    experiment_dir = f"{args.outputs}/{experiment_index:03d}-{model_string_name}"
    dist.barrier()
    if coordinator.is_master():
        os.makedirs(experiment_dir, exist_ok=True)
        with open(f"{experiment_dir}/config.txt", "w") as f:
            json.dump(args.__dict__, f, indent=4)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ==============================
    # Initialize Tensorboard
    # ==============================
    if coordinator.is_master():
        tensorboard_dir = f"{experiment_dir}/tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    # ==============================
    # Initialize Booster
    # stage 2 = shards optimizer states + gradients
    # initial scale for gradient scaler (useful for fp16 dynamic scaling)
    # ==============================
    if args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "ddp":
        plugin = TorchDDPPlugin()
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Process Group
    # ==============================
    sp_size = args.sequence_parallel_size
    dp_size = dist.get_world_size() // sp_size
    pg_manager = ProcessGroupManager(dp_size, sp_size, dp_axis=0, sp_axis=1)

    # ======================================================
    # Initialize Model, Objective, Optimizer
    # ======================================================
    # Create VAE encoder
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.use_video:
        # Wrap the VAE in a wrapper that handles video data
        # We use 2d vae from stableai instead of 3d vqvae from videogpt because it has better results
        vae = AutoencoderKLWrapper(vae)
        # Use 3d patch size that is divisible by the input size
        input_size = (args.num_frames, args.image_size, args.image_size)
        for i in range(3):
            assert input_size[i] % vae.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // vae.patch_size[i] for i in range(3)]
    else:
        assert args.history_frames == 0, "History frames are not supported for image data."
        vae = MultiModalVAEWrapper(vae, sat_chn=args.in_dim, radar_chn=args.out_dim)
        input_size = args.image_size // 8

    # Set mixed precision
    if args.mixed_precision == "bf16" and args.plugin != "ddp":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16" and args.plugin != "ddp":
        dtype = torch.float16
    elif args.mixed_precision == "fp32" and args.plugin == "ddp":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown mixed precision {args.mixed_precision}")
    
    # Set vae to the same dtype as the model
    if not args.use_video:
        vae = vae.to(device).to(dtype)

    # Shared model config for two models
    model_config = {
        "input_size": input_size,
        "in_channels": 8,  # 4 sat latent chn + 4 radar latent chn
        "num_classes": args.num_classes,
        "enable_layernorm_kernel": args.enable_layernorm_kernel,
        "enable_modulate_kernel": args.enable_modulate_kernel,
        "sequence_parallel_size": args.sequence_parallel_size,
        "sequence_parallel_type": args.sequence_parallel_type,
        "text_encoder": args.text_encoder,
    }

    if 'MM' in args.model:
        model_config = {**model_config, 't5_text_encoder': args.t5_text_encoder}

    # Create DiT model
    if "DiT" in args.model:
        if "VDiT" in args.model:
            assert args.use_video, "VDiT model requires video data"
        else:
            assert not args.use_video, "DiT model requires image data"
        model_class = DiT_models[args.model] if 'MM' not in args.model else MMDiT_models[args.model]
    elif "Latte" in args.model:
        assert args.use_video, "Latte model requires video data"
        model_class = Latte_models[args.model] if 'MM' not in args.model else MMLatte_models[args.model]
    else:
        raise ValueError(f"Unknown model {args.model}")
    model = (
        model_class(
            enable_flashattn=args.enable_flashattn,
            sequence_parallel_group=pg_manager.sp_group,
            dtype=dtype,
            **model_config,
        )
        .to(device)
        .to(dtype)
    )

    model_numel = get_model_numel(model)
    logger.info(f"Model params: {format_numel_str(model_numel)}")
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create ema and vae model
    # Note that parameter initialization is done within the DiT constructor
    # Create an EMA of the model for use after training
    ema = model_class(**model_config).to(device)
    ema = ema.to(torch.float32)
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # Create diffusion
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")

    # Setup optimizer
    # Train vae wrapper and model parameters
    trainable_params = list(filter(lambda p: p.requires_grad, vae.parameters())) + list(filter(lambda p: p.requires_grad, model.parameters()))
    # We used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper
    optimizer = HybridAdam(trainable_params, lr=args.lr, weight_decay=0, adamw_mode=True)
    # You can use a lr scheduler if you want
    # Recommend if you continue training from a model
    lr_scheduler = None

    # Prepare models for training
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0, sharded=False)
    # important! This enables embedding dropout for classifier-free guidance
    model.train()
    # EMA model should always be in eval mode
    ema.eval()

    # Setup data:
    if args.use_video:
        #dataset = DatasetFromCSV(
        #    args.data_path,
        #    transform=get_transforms_video(args.image_size),
        #    num_frames=args.num_frames,
        #    frame_interval=args.frame_interval,
        #)
        raise NotImplementedError("Video data is not implemented yet.")
    else:
        # master process goes first
        if not coordinator.is_master():
            dist.barrier()
        # Prepare dataset
        datasetbuilder = DatasetBuilder(
            sat_path=args.sat_files_path,
            radar_path=args.radar_files_path,
            start_date=args.start_date,
            end_date=args.end_date,
            max_folders=args.max_folders,
            history_frames=args.history_frames,
            future_frame=args.future_frame,
            refresh_rate=args.refresh_rate,
            coverage_threshold=args.coverage_threshold,
            seed=args.seed
        )
        dataset_pkl_name = "dataset_filelist.pkl"
        dataset_pkl_path = os.path.join(experiment_dir, dataset_pkl_name)
        if args.retrieve_dataset:
            train_files, val_files, _ = datasetbuilder.load_filelist(dataset_pkl_path)
            logger.info(f"Loaded existing dataset from {dataset_pkl_path}")
        else:
            train_files, val_files, _ = datasetbuilder.build_filelist_by_blocks(
                save_dir=experiment_dir,
                file_name=dataset_pkl_name,
                block_size=args.block_size,
                split_ratio=args.split_ratio,
            )
            logger.info(f"Built new dataset to {dataset_pkl_path}")
        
        # Load dataset
        train_dataset = SatelliteDataset(files=train_files, in_dim=args.in_dim, transform=None)
        val_dataset = SatelliteDataset(files=val_files, in_dim=args.in_dim, transform=None)
        logger.info(f"[Train Dataset] Files: {len(train_files)}, Dataset length: {len(train_dataset)}")
        logger.info(f"[Val Dataset]   Files: {len(val_files)}, Dataset length: {len(val_dataset)}")
        if coordinator.is_master():
            dist.barrier()

    train_dataloader = prepare_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        pg_manager=pg_manager,
    )
    use_validation = len(val_dataset) > 0
    if use_validation:
        val_dataloader = prepare_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=args.num_workers,
            pg_manager=pg_manager,
        )
    else:
        val_dataloader = None
        logger.warning("⚠️ Validation set is empty. Skipping validation during training.")

    # Boost model for distributed training
    torch.set_default_dtype(dtype)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=train_dataloader
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(
            booster, model, ema, optimizer, lr_scheduler, args.load, args.sequence_parallel_type
        )
        logger.info(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    # Only shard ema model when using zero2 plugin
    shard_ema = True if args.plugin == "zero2" else False
    if shard_ema:
        model_sharding(ema)

    num_steps_per_epoch = len(train_dataloader)

    logger.info(f"Training for {args.epochs} epochs...")
    # if resume training, set the sampler start index to the correct value
    train_dataloader.sampler.set_start_index(sampler_start_idx)
    for epoch in range(start_epoch, args.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        train_dataloader_iter = iter(train_dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                if args.use_video:
                    #batch = next(train_dataloader_iter)
                    #x = batch["video"].to(device)
                    #y = batch["text"]
                    raise NotImplementedError("Video data is not implemented yet.")
                else:
                    #x, y = next(dataloader_iter)
                    #x = x.to(device)
                    #y = y.to(device)
                    imgs, masks, *_ = next(train_dataloader_iter) #img[B, H, W, in_dim], mask[b, H, W, out_dim], *_
                    imgs = imgs.permute(0, 3, 1, 2).to(device, dtype=dtype)  # (B, H, W, C) -> (B, C, H, W)
                    masks = masks.permute(0, 3, 1, 2).to(device, dtype=dtype)  # (B, H, W, C) -> (B, C, H, W)

                # VAE encode
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    #x = vae.encode(x)
                    if not args.use_video:
                        imgs = vae.encode_sat(imgs)  # (B, C, H, W) -> (B, latent_chn, H/8, W/8)
                        masks = vae.encode_radar(masks)  # (B, C, H, W) -> (B, latent_chn, H/8, W/8)

                # Diffusion
                t = torch.randint(0, diffusion.num_timesteps, (masks.shape[0],), device=device)
                y = torch.zeros(masks.shape[0], dtype=torch.long, device=device)  # Dummy labels for training
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, masks, imgs, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer, sharded=shard_ema)

                # Log loss values:
                all_reduce_mean(loss)
                global_step = epoch * num_steps_per_epoch + step
                pbar.set_postfix({"loss": loss.item(), "step": step, "global_step": global_step})

                # Log to tensorboard
                if coordinator.is_master() and (global_step + 1) % args.log_every == 0:
                    writer.add_scalar("loss", loss.item(), global_step)

                # Save checkpoint
                if args.ckpt_every > 0 and (global_step + 1) % args.ckpt_every == 0:
                    logger.info(f"Saving checkpoint...")
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        args.batch_size,
                        coordinator,
                        experiment_dir,
                        ema_shape_dict,
                        args.sequence_parallel_type,
                        shard_ema,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {experiment_dir}"
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Parse tuple for dim_scales and input_shape
    def parse_int_tuple(s):
        try:
            # Remove brackets, spaces, convert to integers
            return tuple(map(int, s.strip().strip('()').replace(' ', '').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be a string of integers separated by commas, like '1, 2, 3'.")

    # Parse tuple for split_ratio    
    def parse_float_tuple(s):
        try:
            return tuple(map(float, s.strip().strip('()').replace(' ', '').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be a string of numbers separated by commas, like '0.7, 0.1, 0.2'.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()) + list(Latte_models.keys()) + list(MMDiT_models.keys()) + list(MMLatte_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--use_video", action="store_true", help="Use video data instead of images")
    parser.add_argument("--plugin", type=str, default="zero2")
    parser.add_argument("--outputs", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--load", type=str, default=None, help="Path to a checkpoint dir to load")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5_text_encoder", type=str, default="google-t5/t5-small")

    parser.add_argument("--image_size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num_classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gradient clipping value")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")

    parser.add_argument("--enable_modulate_kernel", action="store_true", help="Enable triton modulate kernel")
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size, enable if > 1")
    parser.add_argument("--sequence_parallel_type", type=str)

    # SHRIMP
    parser.add_argument("--sat_files_path", type=str, default="./datasets", help="Path to satellite image data directory")
    parser.add_argument("--radar_files_path", type=str, default="./datasets", help="Path to radar reflectivity image data directory")
    parser.add_argument("--start_date", type=str, default="", help="Start date for dataset selection (e.g., 20210101)")
    parser.add_argument("--end_date", type=str, default="", help="End date for dataset selection (e.g., 20210430)")
    parser.add_argument("--max_folders", type=int, default=None, help="Maximum number of folders (days) to load. Use None to load all")
    parser.add_argument("--history_frames", type=int, default=0, help="Number of past frames to use as input (set as 0 to use the current frame only)")
    parser.add_argument("--future_frame", type=int, default=0, help="Predict which future frame")
    parser.add_argument("--refresh_rate", type=int, default=10, help="Time interval (in minutes) between frames")
    parser.add_argument("--coverage-threshold", default=0.05, type=float, help="Minimum radar reflectivity coverage threshold for selecting a valid frame (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=96, help="Random seed for dataset buiding.")
    parser.add_argument("--block-size", type=int, default=100, help="Number of sat-radar pairs to include per data segment.")
    parser.add_argument("--split-ratio", type=parse_float_tuple, default=(0.7, 0.2, 0.1), help="Train/val/test split ratio (three floats in [0,1] that sum <= 1.0), e.g. 0.7, 0.1, 0.2")
    parser.add_argument("--fixed-test-days", type=lambda s: s.split(","), default=None, help="Comma-separated list of fixed test folders")
    
    # Control parameters for experiments
    parser.add_argument("--retrieve_dataset", action="store_true", help="store_true: no retrieve; store_false: retrieve")
    parser.add_argument("--in_dim", type=int, default=4, help="Input dimension of the model, 4, 6 or more satellite channels")
    parser.add_argument("--out_dim", type=int, default=1, help="Output dimension of the model, 1 radar channel")

    args = parser.parse_args()
    main(args)
