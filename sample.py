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
from opendit.vae.wrapper import AutoencoderKLWrapper

# SHRIMP
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
        vae = MultiModalVAEWrapper(vae, sat_chn=args.in_dim, radar_chn=args.out_dim)
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
        logger.info(f"Loaded existing dataset from {dataset_pkl_path}")
    else:
        _, _, test_files = datasetbuilder.build_filelist_by_blocks(
            save_dir=args.model_path,
            file_name=dataset_pkl_name,
            block_size=args.block_size,
            split_ratio=args.split_ratio,
        )
        logger.info(f"Built new dataset to {dataset_pkl_path}")
    
    # Load dataset
    test_dataset = SatelliteDataset(files=test_files, in_dim=args.in_dim, transform=None)
    logger.info(f"[Test Dataset] Files: {len(test_files)}, Dataset length: {len(test_dataset)}")
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
        img_times_test.append(imgs_time)
        mask_times_test.append(masks_time)

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
        print(imgs.shape, z.shape, model_kwargs)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, imgs, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        
        elapsed = time.time() - start_time
        test_loop.set_postfix(time=f"{elapsed:.2f}s")
        #if idx >= 10:
        #    logger.info("Sampling break")
        #    break

        # Save and display images:
        if args.use_video:
            samples = vae.decode(samples)
            save_sample(samples)
        else:
            samples = vae.decode(samples / 0.18215).sample
            #save_image(samples.mean(dim=1, keepdim=True), "sample.pdf", nrow=4, normalize=True, value_range=(-1, 1))  # Save mean image
            all_samples.append(samples.cpu())
    
    all_imgs = torch.cat(all_imgs, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()
    all_samples = torch.cat(all_samples, dim=0).numpy()
    all_img_times = torch.cat(all_img_times, dim=0).numpy()
    all_mask_times = torch.cat(all_mask_times, dim=0).numpy()
    np.save(os.path.join(args.results, args.label, f'sats_{args.in_dim-1}.npy'), all_imgs)
    np.save(os.path.join(args.results, args.label, 'reals.npy'), all_masks)
    np.save(os.path.join(args.results, args.label, 'doutputs.npy'), all_samples)
    np.save(os.path.join(args.results, args.label, 'sat_times.npy'), all_img_times)
    np.save(os.path.join(args.results, args.label, 'real_times.npy'), all_mask_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()) + list(Latte_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image_size", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--outputs", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--sat_files_path", type=str, default="./datasets", help="Path to the sat dataset")
    parser.add_argument("--radar_files_path", type=str, default="./datasets", help="Path to the radar dataset")
    parser.add_argument("--start_date", type=str, default="20000101", help="Set dataset start date")
    parser.add_argument("--end_date", type=str, default="20251231", help="Set dataset end date")
    parser.add_argument("--max_folders", type=int, default=None, help="Set dataset max folders")
    parser.add_argument("--history_frames", type=int, default=0, help="Number of history frames")
    parser.add_argument("--future_frame", type=int, default=0, help="Predict which future frame")
    parser.add_argument("--refresh_rate", type=int, default=10, help="Interval of frames")
    parser.add_argument("--retrieve_dataset", action="store_true", help="store_true: no retrieve; store_false: retrieve")
    parser.add_argument("--in_dim", type=int, default=5, help="Input dimension of the model, 4, 6 or more satellite channels, 1 radar channel")
    parser.add_argument("--out_dim", type=int, default=1, help="Output dimension of the model, 1 radar channel")

    args = parser.parse_args()
    main(args)
