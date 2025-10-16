import os
import re
import glob
import pickle
import random
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import logging

logger = logging.getLogger(__name__)

# Prepare dataset
class DatasetBuilder:
    def __init__(self, sat_path, radar_path, start_date="", end_date="", max_folders=None,
                 history_frames=0, future_frame=0, refresh_rate=10,  coverage_threshold=0.05, seed=210):
        self.sat_path = sat_path
        self.radar_path = radar_path
        self.start_date = start_date
        self.end_date = end_date
        self.max_folders = max_folders
        self.history_frames = history_frames
        self.future_frame = future_frame
        self.refresh_rate = refresh_rate
        self.coverage_threshold = coverage_threshold  # Filter dataset by reflectivity coverage
        self.seed = seed

    def extract_time(self, filename):
        match = re.search(r"_(\d{8})_(\d{6})(?:_|\.)", filename)
        return match.group(1) + match.group(2) if match else None
    
    def get_common_folders(self):
        sat_folders = {entry.name for entry in os.scandir(self.sat_path) if entry.is_dir()}
        radar_folders = {entry.name for entry in os.scandir(self.radar_path) if entry.is_dir()}
        common_folders = sorted(sat_folders & radar_folders)
        if self.start_date:
            common_folders = [d for d in common_folders if d >= self.start_date]
        if self.end_date:
            common_folders = [d for d in common_folders if d <= self.end_date]
        if self.max_folders:
            common_folders = common_folders[:self.max_folders]
        return common_folders
    
    def is_radar_sparse(self, radar_file, precip_thr=0.0, conv_thr=33.0):
        """
        Return True if the radar scene is sparse (filtered out), False if valid.

        Logic overview:
        1) Compute precipitation, convective, and stratiform area fractions (PAF/CAF/SAF)
        2) Apply thresholds to filter scenes with insufficient precipitation coverage
        3) Retain cases “popcorn convection” cases
        """

        # --- 1. Early exit if filtering is disabled ---
        if self.coverage_threshold == 0.0:
            return False
        
        # --- 2. Load radar reflectivity array ---
        with xr.open_dataset(radar_file, engine="netcdf4") as dset_radar:
            if "DBZH" in dset_radar:
                radar = dset_radar.DBZH.values
            elif "reflectivity" in dset_radar:
                radar = dset_radar.reflectivity.values
            else:
                return True

        # --- 3. Compute area fractions (PAF/CAF/SAF) ------------------------------
        #   PAF = (pixels > precip_thr) / total pixels (including NaN)
        #   CAF = (pixels > conv_thr) / total pixels (including NaN)
        #   SAF = (pixels <= conv_thr) / total pixels (including NaN)
        radar_flat  = radar.ravel()
        total_pixels = radar_flat.size
        valid = ~np.isnan(radar_flat)

        precip_mask = (radar_flat > precip_thr) & valid
        conv_mask   = (radar_flat > conv_thr) & valid
        strat_mask  = (radar_flat <= conv_thr) & valid

        paf = precip_mask.sum() / total_pixels
        caf = conv_mask.sum()   / total_pixels
        saf = strat_mask.sum()  / total_pixels

        # --- 4. Popcorn convection exception ---
        # Keep cases with low SAF scores but CAF above 1%
        if (saf < 0.05) and (caf >= 0.01):
            return False

        # --- 5. Sparse-scene filtering ---
        # Remove cases with CAF lower than 0.5% or SAF lower than 5%
        is_sparse = (paf < self.coverage_threshold) or (caf < 0.005) or (saf < 0.05)
        return is_sparse
        
    def get_paired_files_from_folders(self, folders, history_frames=0, future_frame=0, refresh_rate=10, return_time=False):
        paired_files = []
        for folder in folders:
            # Check folder path
            sat_folder_path = os.path.join(self.sat_path, folder)
            radar_folder_path = os.path.join(self.radar_path, folder)
            if not os.path.isdir(sat_folder_path) or not os.path.isdir(radar_folder_path):
                logger.info(f"Skipping folder '{folder}': one of the directories is missing!")
                continue
            # Get file path
            sat_files = sorted(glob.glob(os.path.join(sat_folder_path, "*.nc")))
            radar_files = sorted(glob.glob(os.path.join(radar_folder_path, "*.nc")))
            # extract time, store in dic
            sat_dict = {self.extract_time(os.path.basename(f)): f for f in sat_files if self.extract_time(os.path.basename(f))}
            radar_dict = {self.extract_time(os.path.basename(f)): f for f in radar_files if self.extract_time(os.path.basename(f))}
            # Pair files
            sat_times = sorted(sat_dict.keys())
            for t0 in sat_times:
                try:
                    t0_dt = datetime.strptime(t0, "%Y%m%d%H%M%S")
                    hist_times = [(t0_dt - timedelta(minutes=refresh_rate * i)).strftime("%Y%m%d%H%M%S") for i in reversed(range(history_frames+1))]
                    if not all(ht in sat_dict for ht in hist_times):
                        continue
                    sat_files_seq = [sat_dict[ht] for ht in hist_times]
                    # One future radar time at T0 + N * refresh_rate
                    target_radar_time = (t0_dt + timedelta(minutes=refresh_rate * future_frame)).strftime("%Y%m%d%H%M%S")
                    if target_radar_time not in radar_dict:
                        continue
                    radar_file = radar_dict[target_radar_time]
                    if self.is_radar_sparse(radar_file):
                        continue
                    if return_time:
                        # anchor_ts: target_radar_time or t0_dt.
                        anchor_ts = datetime.strptime(target_radar_time, "%Y%m%d%H%M%S")
                        paired_files.append((anchor_ts, sat_files_seq, [radar_file]))
                    else:
                        paired_files.append((sat_files_seq, [radar_file]))
                except ValueError:
                    logger.info(f"Failed to parse time in file: {t0}")
                    continue
        if return_time:
            logger.info(f"Matched {len(paired_files)} sequence pairs (with time keys) "
                        f"with {history_frames} history frames and a future frame of "
                        f"{future_frame * refresh_rate} minutes, refresh rate={refresh_rate} minutes.")
        else:
            logger.info(f"Matched {len(paired_files)} sequence pairs with {history_frames} history frames "
                        f"and a future frame of {future_frame * refresh_rate} minutes, refresh rate={refresh_rate} minutes.")
        return paired_files

    def build_filelist_by_days(self, save_dir, file_name="dataset_filelist.pkl", split_ratio=(0.7, 0.1, 0.2), fixed_test_days=None):
        random.seed(self.seed)
        day_folders = self.get_common_folders()
        if fixed_test_days is not None:
            fixed_test_days = set(str(day) for day in fixed_test_days)  # ensure all are strings
            test_folders = [d for d in day_folders if os.path.basename(d) in fixed_test_days]
            remaining_folders = [d for d in day_folders if os.path.basename(d) not in fixed_test_days]
            
            random.shuffle(remaining_folders)
            total_remaining = len(remaining_folders)
            train_days = round(split_ratio[0] / (split_ratio[0] + split_ratio[1]) * total_remaining)
            val_days = round(split_ratio[1] / (split_ratio[0] + split_ratio[1]) * total_remaining)
            
            train_folders = remaining_folders[:train_days]
            val_folders = remaining_folders[train_days:train_days+val_days]
        else:
            random.shuffle(day_folders)
            total_days = len(day_folders)
            train_days = round(split_ratio[0]*total_days)
            val_days = round(split_ratio[1]*total_days)
            test_days = round(split_ratio[2]*total_days)

            train_folders = day_folders[:train_days]
            val_folders = day_folders[train_days:train_days+val_days]
            test_folders = day_folders[train_days+val_days:train_days+val_days+test_days]

        train_files = self.get_paired_files_from_folders(train_folders, self.history_frames, self.future_frame, self.refresh_rate)
        val_files = self.get_paired_files_from_folders(val_folders, self.history_frames, self.future_frame, self.refresh_rate)
        test_files = self.get_paired_files_from_folders(test_folders, self.history_frames, self.future_frame, self.refresh_rate)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)
        logger.info(f"Saved dataset to: {save_path}")
        logger.info("Total paired files: %d | Total days: %d", len(train_files)+len(val_files)+len(test_files), len(day_folders))
        logger.info("Split (days): train=%d, val=%d, test=%d", len(train_folders), len(val_folders), len(test_folders))
        logger.info("Split (files):  train=%d, val=%d, test=%d", len(train_files), len(val_files), len(test_files))
        return train_files, val_files, test_files

    def build_filelist_by_blocks(self, save_dir, file_name="dataset_filelist.pkl", block_size=100, split_ratio=(0.7, 0.1, 0.2), drop_last=False):
        """
        Process:
        - Collect all paired samples from day_folders (including time keys)
        - Sort by time
        - Split into blocks of block_size (sorted by time within each block)
        - Randomly partition into train/val/test sets at the block level using a random seed (without reshuffling within blocks)
        """
        random.seed(self.seed)

        # 1) Retrieve day folders
        day_folders = self.get_common_folders()

        # 2) Generate a full sample (with timestamp key) for global time sorting
        files_with_time = self.get_paired_files_from_folders(
            day_folders,
            self.history_frames,
            self.future_frame,
            self.refresh_rate,
            return_time=True
        )
        if not files_with_time:
            raise RuntimeError("No paired files found after filtering. Please check paths/thresholds.")

        # 3) Sort by timestamp
        files_with_time.sort(key=lambda x: x[0])  # x = (anchor_ts, sat_seq, [radar])

        # 4) Slicing by blocks (preserving temporal order within each block)
        blocks = [files_with_time[i:i+block_size] for i in range(0, len(files_with_time), block_size)]
        if drop_last and len(blocks) > 0 and len(blocks[-1]) < block_size:
            blocks = blocks[:-1]

        total_blocks = len(blocks)
        if total_blocks == 0:
            raise RuntimeError("No blocks formed. Try smaller block_size or drop_last=False.")

        # 5) Perform random partitioning at the block level (without shuffling within blocks)
        indices = list(range(total_blocks))
        random.shuffle(indices)

        n_train = round(split_ratio[0] * total_blocks)
        n_val   = round(split_ratio[1] * total_blocks)
        n_test  = round(split_ratio[2] * total_blocks)

        train_block_idx = indices[:n_train]
        val_block_idx   = indices[n_train:n_train+n_val]
        test_block_idx  = indices[n_train+n_val:n_train+n_val+n_test]

        # 6) Expand the block to obtain the final sample list (remove the timestamp key, retaining only (sat_seq, [radar])).
        def flatten_blocks(block_idx_list):
            out = []
            for bi in block_idx_list:
                # Block-level retention time sequence (previously sorted by time)
                for (ts, sat_seq, radar_list) in blocks[bi]:
                    out.append((sat_seq, radar_list))
            return out

        train_files = flatten_blocks(train_block_idx)
        val_files   = flatten_blocks(val_block_idx)
        test_files  = flatten_blocks(test_block_idx)

        # 7) Save to file
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)

        logger.info("Saved block-based dataset to: %s", save_path)
        logger.info("Total paired files: %d | Block size: %d | Total blocks: %d", len(files_with_time), block_size, total_blocks)
        logger.info("Split (blocks): train=%d, val=%d, test=%d", len(train_block_idx), len(val_block_idx), len(test_block_idx))
        logger.info("Split (files):  train=%d, val=%d, test=%d", len(train_files), len(val_files), len(test_files))

        return train_files, val_files, test_files

    def load_filelist(self, path):
        with open(path, 'rb') as f:
            logger.info(f"Loaded dataset from: {path}")
            return pickle.load(f)