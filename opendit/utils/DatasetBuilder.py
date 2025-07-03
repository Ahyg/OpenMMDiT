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
        sat_folders = set(os.listdir(self.sat_path))
        radar_folders = set(os.listdir(self.radar_path))
        common_folders = sorted(list(sat_folders & radar_folders))
        if self.start_date:
            common_folders = [d for d in common_folders if d >= self.start_date]
        if self.end_date:
            common_folders = [d for d in common_folders if d <= self.end_date]
        if self.max_folders:
            common_folders = common_folders[:self.max_folders]
        return common_folders
    
    def is_rainfall_sparse(self, radar_file):
        dset_radar = xr.open_dataset(radar_file, engine='netcdf4')
        if "DBZH" in dset_radar:
            radar = dset_radar.DBZH.values
            dset_radar.close()
            total_pixels = radar.size
            nonzero_pixels = np.count_nonzero(~np.isnan(radar) & (radar != 0))
            return (nonzero_pixels / total_pixels) < self.coverage_threshold
        dset_radar.close()
        return True
        
    def get_paired_files_from_folders(self, folders, history_frames=0, future_frame=0, refresh_rate=10):
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
                    if self.is_rainfall_sparse(radar_file):
                        continue
                    paired_files.append((sat_files_seq, [radar_file]))
                except ValueError:
                    logger.info(f"Failed to parse time in file: {t0}")
                    continue
        logger.info(f"Matched {len(paired_files)} sequence pairs with {history_frames} history frames and a future frame of {future_frame * refresh_rate} minutes, refresh rate={refresh_rate} minutes.")
        return paired_files

    def build_filelist(self, save_dir, file_name="dataset_files.pkl", split_ratio=(0.7, 0.1, 0.2)):
        random.seed(self.seed)
        day_folders = self.get_common_folders()
        random.shuffle(day_folders)
        total_days = len(day_folders)
        train_days = round(split_ratio[0]*total_days)
        val_days = round(split_ratio[1]*total_days)
        #test_days = total_days - train_days - val_days

        train_folders = day_folders[:train_days]
        val_folders = day_folders[train_days:train_days+val_days]
        test_folders = day_folders[train_days+val_days:]

        train_files = self.get_paired_files_from_folders(train_folders, self.history_frames, self.future_frame, self.refresh_rate)
        val_files = self.get_paired_files_from_folders(val_folders, self.history_frames, self.future_frame, self.refresh_rate)
        test_files = self.get_paired_files_from_folders(test_folders, self.history_frames, self.future_frame, self.refresh_rate)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)
        logger.info(f"Saved dataset to: {save_path}")
        return train_files, val_files, test_files

    def load_filelist(self, path):
        with open(path, 'rb') as f:
            logger.info(f"Loaded dataset from: {path}")
            return pickle.load(f)
    
# For old dataset
class DatasetBuilder_o:
    def __init__(self, sat_path, radar_path, start_date="", end_date="", max_folders=None,
                 history_frames=0, future_frame=0, refresh_rate=10, coverage_threshold=0.05, seed=210):
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
        sat_folders = set(os.listdir(self.sat_path))
        radar_folders = set(os.listdir(self.radar_path))
        common_folders = sorted(list(sat_folders & radar_folders))
        if self.start_date:
            common_folders = [d for d in common_folders if d >= self.start_date]
        if self.end_date:
            common_folders = [d for d in common_folders if d <= self.end_date]
        if self.max_folders:
            common_folders = common_folders[:self.max_folders]
        return common_folders
    
    def is_radar_sparse(self, radar_file):
        dset_radar = xr.open_dataset(radar_file, engine='netcdf4')
        if "reflectivity" in dset_radar:
            radar = dset_radar.reflectivity.values
            dset_radar.close()
            total_pixels = radar.size
            nonzero_pixels = np.count_nonzero(~np.isnan(radar) & (radar != 0))
            return (nonzero_pixels / total_pixels) < self.coverage_threshold
        dset_radar.close()
        return True
        
    def get_paired_files_from_folders(self, folders, history_frames=0, future_frame=0, refresh_rate=10):
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
                        logger.info("Sparse")
                        continue
                    paired_files.append((sat_files_seq, [radar_file]))
                except ValueError:
                    logger.info(f"Failed to parse time in file: {t0}")
                    continue
        logger.info(f"Matched {len(paired_files)} sequence pairs with {history_frames} history frames and a future frame of {future_frame * refresh_rate} minutes, refresh rate={refresh_rate} minutes.")
        return paired_files

    def build_filelist(self, save_dir, file_name="dataset_files.pkl", split_ratio=(0.7, 0.1, 0.2)):
        random.seed(self.seed)
        day_folders = self.get_common_folders()
        random.shuffle(day_folders)
        total_days = len(day_folders)
        train_days = round(split_ratio[0]*total_days)
        val_days = round(split_ratio[1]*total_days)
        #test_days = total_days - train_days - val_days

        train_folders = day_folders[:train_days]
        val_folders = day_folders[train_days:train_days+val_days]
        test_folders = day_folders[train_days+val_days:]

        train_files = self.get_paired_files_from_folders(train_folders, self.history_frames, self.future_frame, self.refresh_rate)
        val_files = self.get_paired_files_from_folders(val_folders, self.history_frames, self.future_frame, self.refresh_rate)
        test_files = self.get_paired_files_from_folders(test_folders, self.history_frames, self.future_frame, self.refresh_rate)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((train_files, val_files, test_files), f)
        logger.info(f"Saved dataset to: {save_path}")
        return train_files, val_files, test_files

    def load_filelist(self, path):
        with open(path, 'rb') as f:
            logger.info(f"Loaded dataset from: {path}")
            return pickle.load(f)