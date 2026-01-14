import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def extract_features(data_dir="mallorn-astronomical-classification-challenge"):
    train_log = pd.read_csv(os.path.join(data_dir, "train_log.csv"))
    test_log = pd.read_csv(os.path.join(data_dir, "test_log.csv"))
    
    # Extinction coefficients A_lambda / E(B-V) for LSST filters
    R_lambda = {
        'u': 4.81, 'g': 3.64, 'r': 2.70,
        'i': 2.06, 'z': 1.58, 'y': 1.31
    }
    
    # Gaussian Process Kernel: RBF (smooth signal) + WhiteKernel (noise)
    # We use fixed bounds to speed up fitting
    kernel = RBF(length_scale=10.0, length_scale_bounds=(1.0, 1000.0)) + \
             WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 100.0))

    def process_split(split_name, is_train=True):
        prefix = "train" if is_train else "test"
        lc_file = os.path.join(data_dir, split_name, f"{prefix}_full_lightcurves.csv")
        if not os.path.exists(lc_file):
            return []
            
        df_lc = pd.read_csv(lc_file)
        log_df = train_log if is_train else test_log
        
        features_list = []
        
        for obj_id, group in df_lc.groupby('object_id'):
            obj_log = log_df[log_df['object_id'] == obj_id].iloc[0]
            ebv = obj_log['EBV']
            redshift = obj_log['Z']
            
            features = {
                'object_id': obj_id,
                'Z': redshift,
                'EBV': ebv,
            }
            if is_train:
                features['target'] = obj_log['target']
            
            features['n_obs'] = len(group)
            
            # --- Absolute Magnitude Approximation ---
            if redshift > 0:
                dist_mod = 5 * np.log10(4.285e9 * redshift) - 5
            else:
                dist_mod = 0
                
            # Combine all filters for a "global" GP fit (simplified for speed)
            # Normalizing flux to avoid scaling issues
            all_time = group['Time (MJD)'].values
            all_flux = group['Flux'].values
            
            # --- 4. Gaussian Process Fit (Global Smoothness) ---
            # We fit on the filter with the most points (usually 'r' or 'g') to save time
            # or combined. Let's try the 'r' band as it's usually well sampled.
            best_filter = 'r'
            f_group = group[group['Filter'] == best_filter]
            if len(f_group) > 5:
                X = f_group['Time (MJD)'].values.reshape(-1, 1)
                y = f_group['Flux'].values
                # Normalize inputs for stable GP
                y_mean = np.mean(y)
                y_std = np.std(y) + 1e-6
                y_norm = (y - y_mean) / y_std
                
                try:
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=0, random_state=42)
                    gp.fit(X, y_norm)
                    features['gp_length_scale'] = gp.kernel_.k1.length_scale
                    features['gp_noise'] = gp.kernel_.k2.noise_level
                except:
                    features['gp_length_scale'] = np.nan
                    features['gp_noise'] = np.nan
            else:
                features['gp_length_scale'] = np.nan
                features['gp_noise'] = np.nan

            filter_fluxes = {}

            for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
                f_group = group[group['Filter'] == filt]
                if f_group.empty:
                    for stat in ['mean', 'std', 'max', 'min', 'w_mean', 'skew', 'kurt', 
                                 'peak_to_mean', 'duration', 'peak_time', 'decay_slope', 'abs_mag_max']:
                        features[f'{filt}_{stat}'] = np.nan
                    features[f'{filt}_n_obs'] = 0
                    continue
                
                # De-extinction
                corr_factor = 10**(0.4 * R_lambda[filt] * ebv)
                flux = f_group['Flux'] * corr_factor
                flux_err = f_group['Flux_err'] * corr_factor
                time = f_group['Time (MJD)']
                
                filter_fluxes[filt] = flux.values
                
                # Basic Stats
                features[f'{filt}_mean'] = flux.mean()
                features[f'{filt}_std'] = flux.std()
                features[f'{filt}_max'] = flux.max()
                features[f'{filt}_min'] = flux.min()
                features[f'{filt}_n_obs'] = len(f_group)
                
                # Weighted Mean
                weights = 1.0 / (flux_err**2 + 1e-8)
                features[f'{filt}_w_mean'] = np.average(flux, weights=weights)
                
                # Shape Stats
                if len(flux) > 2:
                    features[f'{filt}_skew'] = skew(flux)
                    features[f'{filt}_kurt'] = kurtosis(flux)
                else:
                    features[f'{filt}_skew'] = 0
                    features[f'{filt}_kurt'] = -3
                
                features[f'{filt}_peak_to_mean'] = flux.max() / (flux.mean() + 1e-5)
                
                # Time-based features
                if len(f_group) > 1:
                    features[f'{filt}_duration'] = time.max() - time.min()
                    peak_idx = flux.idxmax()
                    t_peak = f_group.loc[peak_idx, 'Time (MJD)']
                    features[f'{filt}_peak_time'] = (t_peak - time.min()) / features[f'{filt}_duration']
                    
                    # --- Power Law Decay Fit ---
                    post_peak = f_group[f_group['Time (MJD)'] > t_peak]
                    if len(post_peak) >= 4:
                        t_decay = post_peak['Time (MJD)'] - t_peak
                        flux_decay = post_peak['Flux'] * corr_factor
                        valid_mask = (flux_decay > 0) & (t_decay > 0)
                        if valid_mask.sum() >= 3:
                            try:
                                log_t = np.log(t_decay[valid_mask])
                                log_f = np.log(flux_decay[valid_mask])
                                slope, intercept = np.polyfit(log_t, log_f, 1)
                                features[f'{filt}_decay_slope'] = -slope
                            except:
                                features[f'{filt}_decay_slope'] = np.nan
                        else:
                             features[f'{filt}_decay_slope'] = np.nan
                    else:
                        features[f'{filt}_decay_slope'] = np.nan
                        
                else:
                    features[f'{filt}_duration'] = 0
                    features[f'{filt}_peak_time'] = 0
                    features[f'{filt}_decay_slope'] = np.nan
                
                # --- Absolute Magnitude (Max) ---
                if features[f'{filt}_max'] > 0:
                    m_app = -2.5 * np.log10(features[f'{filt}_max']) + 23.9
                    features[f'{filt}_abs_mag_max'] = m_app - dist_mod
                else:
                    features[f'{filt}_abs_mag_max'] = np.nan
            
            # --- Color Features ---
            for f1, f2 in [('g', 'r'), ('r', 'i')]:
                if f1 in filter_fluxes and f2 in filter_fluxes:
                    if features[f'{f2}_mean'] > 0:
                        features[f'{f1}_{f2}_ratio'] = features[f'{f1}_mean'] / features[f'{f2}_mean']
                    else:
                        features[f'{f1}_{f2}_ratio'] = np.nan
                        
                    if features[f'{f2}_std'] > 0:
                        features[f'{f1}_{f2}_std_ratio'] = features[f'{f1}_std'] / features[f'{f2}_std']
                    else:
                        features[f'{f1}_{f2}_std_ratio'] = np.nan
                else:
                    features[f'{f1}_{f2}_ratio'] = np.nan
                    features[f'{f1}_{f2}_std_ratio'] = np.nan

            features_list.append(features)
        return features_list

    print("Extracting training features...")
    train_features = []
    splits = sorted(train_log['split'].unique())
    for i, split in enumerate(splits):
        print(f"Processing split {i+1}/{len(splits)}: {split}")
        train_features.extend(process_split(split, is_train=True))
        
    print("Extracting test features...")
    test_features = []
    splits = sorted(test_log['split'].unique())
    for i, split in enumerate(splits):
        print(f"Processing split {i+1}/{len(splits)}: {split}")
        test_features.extend(process_split(split, is_train=False))
        
    df_train = pd.DataFrame(train_features)
    df_test = pd.DataFrame(test_features)
    
    df_train.to_csv("train_features.csv", index=False)
    df_test.to_csv("test_features.csv", index=False)
    print("Features saved to train_features.csv and test_features.csv")

if __name__ == "__main__":
    extract_features()