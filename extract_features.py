import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis

def extract_features(data_dir="mallorn-astronomical-classification-challenge"):
    train_log = pd.read_csv(os.path.join(data_dir, "train_log.csv"))
    test_log = pd.read_csv(os.path.join(data_dir, "test_log.csv"))
    
    # Extinction coefficients A_lambda / E(B-V) for LSST filters
    R_lambda = {
        'u': 4.81, 'g': 3.64, 'r': 2.70,
        'i': 2.06, 'z': 1.58, 'y': 1.31
    }
    
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
            
            features = {
                'object_id': obj_id,
                'Z': obj_log['Z'],
                'EBV': ebv,
            }
            if is_train:
                features['target'] = obj_log['target']
            
            features['n_obs'] = len(group)
            
            for filt in ['u', 'g', 'r', 'i', 'z', 'y']:
                f_group = group[group['Filter'] == filt]
                if f_group.empty:
                    for stat in ['mean', 'std', 'max', 'min', 'w_mean', 'skew', 'kurt', 'peak_to_mean', 'duration', 'peak_time']:
                        features[f'{filt}_{stat}'] = np.nan
                    features[f'{filt}_n_obs'] = 0
                    continue
                
                # De-extinction
                corr_factor = 10**(0.4 * R_lambda[filt] * ebv)
                flux = f_group['Flux'] * corr_factor
                flux_err = f_group['Flux_err'] * corr_factor
                
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
                    features[f'{filt}_duration'] = f_group['Time (MJD)'].max() - f_group['Time (MJD)'].min()
                    peak_idx = flux.idxmax()
                    t_min = f_group['Time (MJD)'].min()
                    t_range = features[f'{filt}_duration']
                    t_peak = f_group.loc[peak_idx, 'Time (MJD)']
                    features[f'{filt}_peak_time'] = (t_peak - t_min) / t_range if t_range > 0 else 0
                else:
                    features[f'{filt}_duration'] = 0
                    features[f'{filt}_peak_time'] = 0
            
            # Cross-filter features (Colors/Ratios)
            for f1, f2 in [('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'g'), ('z', 'y')]:
                val1 = features.get(f'{f1}_mean', np.nan)
                val2 = features.get(f'{f2}_mean', np.nan)
                
                if not np.isnan(val1) and not np.isnan(val2) and val2 != 0:
                     features[f'{f1}_{f2}_ratio'] = val1 / val2
                else:
                     features[f'{f1}_{f2}_ratio'] = np.nan

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