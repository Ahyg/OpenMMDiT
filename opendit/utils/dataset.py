import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

# For Baseu
class SatelliteDataset_baseu(Dataset):
    def __init__(self, files, in_dim=4, transform=None):
        super().__init__()
        self.files = files
        self.in_dim = in_dim
        self.transform = transform

    def scale_sat_img(self, img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        if img.shape[-1] == 6:
            img[..., 5] = img[..., 5] / 2.0 + 0.5
        np.clip(img, 0.0, 1.0, out=img)
        return img  # 0~1

    def scale_radar_img(self, mask):
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        np.clip(mask, 0.0, 60.0, out=mask)
        normalized_mask = mask / 60
        return normalized_mask  # 0~1

    def read_data(self, sat_files, radar_files):
        # Read satellite data
        sats = []
        sat_times = []
        for sat_file in sat_files:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    if 'satcomp' in sat_dset and 'normed' in sat_dset:
                        satcomp = sat_dset['satcomp'].values  # shape: (H, W, C1)
                        normed = sat_dset['normed'].values    # shape: (H, W, C=8)
                        normed_ltng = normed[:, :, 3:4]  # shape: (H, W, 1)
                        if self.in_dim == 4:  # Sat
                            sat = np.concatenate([satcomp, normed_ltng], axis=-1)  # shape: (H, W, C1+1)
                        elif self.in_dim == 6:  # Sat (include sun)
                            sun = normed[:, :, 6:8]
                            sat = np.concatenate([satcomp, normed_ltng, sun], axis=-1)  # shape: (H, W, C1+1+2)
                    elif 'satellite' in sat_dset:
                        sat = sat_dset['satellite'].values  # (H, W, 4)
                    else:
                        raise ValueError("Unrecognized satellite file format.")
                    if 'time' in sat_dset:
                        sat_times.append(sat_dset['time'].values)
                    elif 'date' in sat_dset:
                        sat_times.append(sat_dset['date'].values)
                    else:  
                        raise KeyError("Missing time/date variable in satellite dataset.")
                    sat = self.scale_sat_img(sat)
                    sats.append(sat)
                except KeyError as e:
                    raise ValueError(f"Missing expected satellite variable: {e}")
        concated_sat = np.concatenate(sats, axis=-1)  # shape: (H, W, (C1+x)*(history_frames+1))
        sat_time = sat_times[-1]  # T0
        # Read radar data
        radar_file = radar_files[0]
        with xr.open_dataset(radar_file, engine='netcdf4') as radar_dset:
            try:
                if "DBZH" in radar_dset:
                    radar = radar_dset["DBZH"].values  # shape: (H, W)
                elif "reflectivity" in radar_dset:
                    radar = radar_dset["reflectivity"].values
                else:
                    raise ValueError("Unrecognized radar file format.")
                if 'time' in radar_dset:
                    radar_time = radar_dset['time'].values
                elif 'date' in radar_dset:
                    radar_time = radar_dset['date'].values
                else:
                    raise KeyError("Missing time/date variable in radar dataset.")
                if radar.ndim == 2:
                    radar = np.expand_dims(radar, axis=-1)  # make it (H, W, 1)
                radar = self.scale_radar_img(radar)
            except KeyError as e:
                raise ValueError(f"Missing expected radar variable: {e}")
        return concated_sat, radar, sat_time, radar_time

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sat_files, radar_files = self.files[idx]
        img, mask, sat_time, radar_time = self.read_data(sat_files, radar_files)

        #img = torch.from_numpy(img).float()
        #mask = torch.from_numpy(mask).float()
        sat_time = sat_time.astype('datetime64[s]').astype('int64')
        radar_time = radar_time.astype('datetime64[s]').astype('int64')

        return img, mask, sat_time, radar_time
    
# For Diff
class SatelliteDataset(Dataset):
    def __init__(self, files, in_dim=4, transform=None):
        super().__init__()
        self.files = files
        self.in_dim = in_dim
        self.transform = transform

    def scale_sat_img(self, img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        if img.shape[-1] == 6:
            img[..., 5] = img[..., 5] / 2.0 + 0.5
        np.clip(img, 0.0, 1.0, out=img)
        scaled_img = 2 * img - 1
        return scaled_img  # -1 ~ 1

    def scale_radar_img(self, mask):
        mask = np.nan_to_num(mask, nan=0.0, copy=False)
        np.clip(mask, 0.0, 60.0, out=mask)
        normalized_mask = mask / 30.0 - 1
        return normalized_mask  # -1 ~ 1

    def read_data(self, sat_files, radar_files):
        # Read satellite data
        sats = []
        sat_times = []
        for sat_file in sat_files:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    if 'satcomp' in sat_dset and 'normed' in sat_dset:
                        satcomp = sat_dset['satcomp'].values  # shape: (H, W, C1)
                        normed = sat_dset['normed'].values    # shape: (H, W, C=8)
                        normed_ltng = normed[:, :, 3:4]  # shape: (H, W, 1)
                        if self.in_dim == 4:  # Sat
                            sat = np.concatenate([satcomp, normed_ltng], axis=-1)  # shape: (H, W, C1+1)
                        elif self.in_dim == 6:  # Sat (include sun)
                            sun = normed[:, :, 6:8]
                            sat = np.concatenate([satcomp, normed_ltng, sun], axis=-1)  # shape: (H, W, C1+1+2)
                    elif 'satellite' in sat_dset:
                        sat = sat_dset['satellite'].values  # (H, W, 4)
                    else:
                        raise ValueError("Unrecognized satellite file format.")
                    if 'time' in sat_dset:
                        sat_times.append(sat_dset['time'].values)
                    elif 'date' in sat_dset:
                        sat_times.append(sat_dset['date'].values)
                    else:  
                        raise KeyError("Missing time/date variable in satellite dataset.")
                    sat = self.scale_sat_img(sat)
                    sats.append(sat)
                except KeyError as e:
                    raise ValueError(f"Missing expected satellite variable: {e}")
        concated_sat = np.concatenate(sats, axis=-1)  # shape: (H, W, (C1+x)*(history_frames+1))
        sat_time = sat_times[-1]  # T0
        # Read radar data
        radar_file = radar_files[0]
        with xr.open_dataset(radar_file, engine='netcdf4') as radar_dset:
            try:
                if "DBZH" in radar_dset:
                    radar = radar_dset["DBZH"].values  # shape: (H, W)
                elif "reflectivity" in radar_dset:
                    radar = radar_dset["reflectivity"].values
                else:
                    raise ValueError("Unrecognized radar file format.")
                if 'time' in radar_dset:
                    radar_time = radar_dset['time'].values
                elif 'date' in radar_dset:
                    radar_time = radar_dset['date'].values
                else:
                    raise KeyError("Missing time/date variable in radar dataset.")
                if radar.ndim == 2:
                    radar = np.expand_dims(radar, axis=-1)  # make it (H, W, 1)
                radar = self.scale_radar_img(radar)
            except KeyError as e:
                raise ValueError(f"Missing expected radar variable: {e}")
        return concated_sat, radar, sat_time, radar_time

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sat_files, radar_files = self.files[idx]
        img, mask, sat_time, radar_time = self.read_data(sat_files, radar_files)

        #img = torch.from_numpy(img).float()
        #mask = torch.from_numpy(mask).float()
        sat_time = sat_time.astype('datetime64[s]').astype('int64')
        radar_time = radar_time.astype('datetime64[s]').astype('int64')

        return img, mask, sat_time, radar_time
    
# For whole aus data    
class SatelliteDataset_w(Dataset):
    def __init__(self, files, in_dim=5, transform=None):
        super().__init__()
        self.files = files
        self.in_dim = in_dim
        self.transform = transform

    def scale_sat_img(self, img):
        img = np.nan_to_num(img, nan=0.0, copy=False)
        if img.shape[-1] == 6:
            img[..., 5] = img[..., 5] / 2.0 + 0.5
        np.clip(img, 0.0, 1.0, out=img)
        scaled_img = 2 * img - 1
        return scaled_img  # -1 ~ 1

    def read_data(self, sat_files):
        # Read satellite data
        sats = []
        sat_times = []
        for sat_file in sat_files:
            with xr.open_dataset(sat_file, engine='netcdf4') as sat_dset:
                try:
                    satcomp = sat_dset['satcomp'].values  # shape: (H, W, C1)
                    normed = sat_dset['normed'].values    # shape: (H, W, C=8)
                    normed_ltng = normed[:, :, 3:4]  # shape: (H, W, 1)
                    if self.in_dim == 5:  # Sat + radar
                        sat = np.concatenate([satcomp, normed_ltng], axis=-1)  # shape: (H, W, C1+1)
                    elif self.in_dim == 7:  # Sat (include sun) + radar
                        sun = normed[:, :, 6:8]
                        sat = np.concatenate([satcomp, normed_ltng, sun], axis=-1)  # shape: (H, W, C1+1+2)
                    sat = self.scale_sat_img(sat)
                    sats.append(sat)
                    sat_times.append(sat_dset['time'].values)
                except KeyError as e:
                    raise ValueError(f"Missing expected satellite variable: {e}")
        concated_sat = np.concatenate(sats, axis=-1)  # shape: (H, W, (C1+x)*(history_frames+1))
        sat_time = sat_times[-1]  # T0
        # No radar data
        radar = -1 * np.ones_like(sat_times, dtype=np.float32)
        if radar.ndim == 2:
            radar = np.expand_dims(radar, axis=-1)  # make it (H, W, 1)
        radar_time = sat_time
        return concated_sat, radar, sat_time, radar_time
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sat_files, _ = self.files[idx]
        img, mask, sat_time, radar_time = self.read_data(sat_files)

        sat_time = sat_time.astype('datetime64[s]').astype('int64')
        radar_time = radar_time.astype('datetime64[s]').astype('int64')
        
        return img, mask, sat_time, radar_time