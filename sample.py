# Modified from Meta DiT: https://github.com/facebookresearch/DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import argparse

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from opendit.diffusion import create_diffusion
from opendit.models.dit import DiT_models
from opendit.models.latte import Latte_models
from opendit.utils.download import find_model
from opendit.vae.reconstruct import save_sample
from opendit.vae.wrapper import AutoencoderKLWrapper, MultiModalVAEWrapper

# SHRIMP
from opendit.utils.ckpt_utils import create_logger
from opendit.utils.DatasetBuilder import DatasetBuilder
from opendit.utils.dataset import SatelliteDataset
from opendit.utils.data_utils import prepare_dataloader
from tqdm import tqdm
import time
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    # Load model:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.use_video:
        # Wrap the VAE in a wrapper that handles video data
        # Use 3d patch size that is divisible by the input size
        vae = AutoencoderKLWrapper(vae)
        input_size = (args.num_frames, args.image_size, args.image_size)
        for i in range(3):
            assert input_size[i] % vae.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // vae.patch_size[i] for i in range(3)]
    else:
        assert args.history_frames == 0, "History frames are not supported for image data."
        vae = MultiModalVAEWrapper(vae, sat_chn=args.in_dim, radar_chn=args.out_dim).to(device)
        input_size = args.image_size // 8


    dtype = torch.float32
    if "DiT" in args.model:
        if "VDiT" in args.model:
            assert args.use_video, "VDiT model requires video data"
        else:
            assert not args.use_video, "DiT model requires image data"
        model_class = DiT_models[args.model]
    elif "Latte" in args.model:
        assert args.use_video, "Latte model requires video data"
        model_class = Latte_models[args.model]
    else:
        raise ValueError(f"Unknown model {args.model}")
    model = (
        model_class(
            input_size=input_size,
            in_channels=8,  # 4 sat latent chn + 4 radar latent chn
            num_classes=args.num_classes,
            enable_flashattn=False,
            enable_layernorm_kernel=False,
            dtype=dtype,
            text_encoder=args.text_encoder,
        )
        .to(device)
        .to(dtype)
    )

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Setup data:
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
    dataset_pkl_path = os.path.join(args.model_path, dataset_pkl_name)
    if args.retrieve_dataset:
        _, _, test_files = datasetbuilder.load_filelist(dataset_pkl_path)
        print(f"Loaded existing dataset from {dataset_pkl_path}")
    else:
        _, _, test_files = datasetbuilder.build_filelist_by_blocks(
            save_dir=args.model_path,
            file_name=dataset_pkl_name,
            block_size=args.block_size,
            split_ratio=args.split_ratio,
        )
        print(f"Built new dataset to {dataset_pkl_path}")
    
    # Load dataset
    test_dataset = SatelliteDataset(files=test_files, in_dim=args.in_dim, transform=None)
    print(f"[Test Dataset] Files: {len(test_files)}, Dataset length: {len(test_dataset)}")
    test_dataloader = prepare_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    doutputs, imgs_test, masks_test, img_times_test, mask_times_test = [], [], [], [], []
    test_loop = tqdm(test_dataloader, desc="Sampling (Test)", total=len(test_dataloader))
    for idx, data in enumerate(test_loop):
        start_time = time.time()

        imgs, masks, img_times, mask_times = data
        
        imgs_test.append(imgs)
        masks_test.append(masks)
        img_times_test.append(img_times)
        mask_times_test.append(mask_times)

        imgs = imgs.permute(0, 3, 1, 2).to(device, dtype=dtype)

        # VAE encode
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            #x = vae.encode(x)
            if not args.use_video:
                imgs = vae.encode_sat(imgs)  # (B, C, H, W) -> (B, latent_chn, H/8, W/8)
        
        # Create sampling noise:
        if args.use_video:
            # Labels to condition the model with (feel free to change):
            class_labels = ["Biking", "Cliff Diving", "Rock Climbing Indoor", "Punch", "TaiChi"]
            n = len(class_labels)
            z = torch.randn(n, vae.out_channels, *input_size, device=device)
            y = class_labels * 2
        else:
            # Labels to condition the model with (feel free to change):
            if args.num_classes == 1000:
                class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
            else:
                class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            n = imgs.size(0)
            z = torch.randn(n, 4, input_size, input_size, device=device)
            y = torch.zeros(imgs.shape[0], dtype=torch.long, device=device)
            y_null = torch.zeros(imgs.shape[0], dtype=torch.long, device=device)
            y = torch.cat([y, y_null], 0)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        imgs = torch.cat([imgs, imgs], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, imgs, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        
        elapsed = time.time() - start_time
        test_loop.set_postfix(time=f"{elapsed:.2f}s")
        #if idx >= 10:
        #    print("Sampling break")
        #    break

        # Save and display images:
        if args.use_video:
            samples = vae.decode(samples)
            save_sample(samples)
        else:
            samples = vae.decode_radar(samples)
            #save_image(samples.mean(dim=1, keepdim=True), "sample.pdf", nrow=4, normalize=True, value_range=(-1, 1))  # Save mean image
            samples = samples.permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5
            doutputs.append(samples)
    
    #loaded_model_name = os.path.splitext(os.path.basename(args.load_model))[0]
    loaded_model_name = "DiT-S-2"
    result_dir = os.path.join(args.results, f"{loaded_model_name}")
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, f'doutputs.npy'), np.concatenate(doutputs, axis=0))  # 0~1
    print(f"Test results saved to {os.path.join(result_dir, f'doutputs.npy')}")
    os.makedirs(args.datasets, exist_ok=True)
    np.save(os.path.join(args.datasets, f'sats_{args.in_dim}.npy'), np.concatenate(imgs_test, axis=0))
    np.save(os.path.join(args.datasets, f'sat_times.npy'), np.concatenate(img_times_test, axis=0))
    np.save(os.path.join(args.datasets, f'reals.npy'), np.concatenate(masks_test, axis=0))
    np.save(os.path.join(args.datasets, f'real_times.npy'), np.concatenate(mask_times_test, axis=0))


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
        "--model", type=str, choices=list(DiT_models.keys()) + list(Latte_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image_size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    #parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use_video", action="store_true", help="Use video data instead of images.")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )

    # SHRIMP
    parser.add_argument("--sat_files_path", type=str, default="./datasets", help="Path to satellite image data directory")
    parser.add_argument("--radar_files_path", type=str, default="./datasets", help="Path to radar reflectivity image data directory")
    parser.add_argument("--start_date", type=str, default="", help="Start date for dataset selection (e.g., 20210101)")
    parser.add_argument("--end_date", type=str, default="", help="End date for dataset selection (e.g., 20210430)")
    parser.add_argument("--max_folders", type=int, default=None, help="Maximum number of folders (days) to load. Use None to load all")
    parser.add_argument("--history_frames", type=int, default=0, help="Number of past frames to use as input (set as 0 to use the current frame only)")
    parser.add_argument("--future_frame", type=int, default=0, help="Predict which future frame")
    parser.add_argument("--refresh_rate", type=int, default=10, help="Time interval (in minutes) between frames")
    parser.add_argument("--coverage_threshold", default=0.05, type=float, help="Minimum radar reflectivity coverage threshold for selecting a valid frame (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=96, help="Random seed for dataset buiding.")
    parser.add_argument("--block_size", type=int, default=100, help="Number of sat-radar pairs to include per data segment.")
    parser.add_argument("--split_ratio", type=parse_float_tuple, default=(0.7, 0.2, 0.1), help="Train/val/test split ratio (three floats in [0,1] that sum <= 1.0), e.g. 0.7, 0.1, 0.2")
    parser.add_argument("--fixed_test_days", type=lambda s: s.split(","), default=None, help="Comma-separated list of fixed test folders")
    
    # Control parameters for experiments
    parser.add_argument("--retrieve_dataset", action="store_true", help="store_true: no retrieve; store_false: retrieve")
    parser.add_argument("--in_dim", type=int, default=4, help="Input dimension of the model, 4, 6 or more satellite channels")
    parser.add_argument("--out_dim", type=int, default=1, help="Output dimension of the model, 1 radar channel")
    parser.add_argument("--datasets", type=str, default="", help="Path to cached dataset (npy).")
    parser.add_argument("--model_path", type=str, default="", help="Path to save or load model checkpoints and logs.")
    parser.add_argument("--results", type=str, default="", help="Path to save inference results.")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
