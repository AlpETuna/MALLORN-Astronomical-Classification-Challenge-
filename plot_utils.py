import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_lightcurve(object_id, data_dir="mallorn-astronomical-classification-challenge"):
    # Find which split the object is in
    train_log = pd.read_csv(os.path.join(data_dir, "train_log.csv"))
    test_log = pd.read_csv(os.path.join(data_dir, "test_log.csv"))
    
    log = pd.concat([train_log, test_log])
    obj_info = log[log['object_id'] == object_id]
    
    if obj_info.empty:
        print(f"Object {object_id} not found in logs.")
        return
    
    split = obj_info.iloc[0]['split']
    target = obj_info.iloc[0].get('target', 'Unknown')
    spec_type = obj_info.iloc[0].get('SpecType', 'Unknown')
    ebv = obj_info.iloc[0]['EBV']
    
    # Try to load from train or test
    train_file = os.path.join(data_dir, split, "train_full_lightcurves.csv")
    test_file = os.path.join(data_dir, split, "test_full_lightcurves.csv")
    
    lc = None
    if os.path.exists(train_file):
        df = pd.read_csv(train_file)
        lc = df[df['object_id'] == object_id]
    
    if lc is None or lc.empty:
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            lc = df[df['object_id'] == object_id]
            
    if lc is None or lc.empty:
        print(f"Lightcurve for {object_id} not found in {split}.")
        return

    # De-extinct the flux (Simplified version, real one might need filter-specific extinction)
    # The description says: "This value can be used to de-extinct the flux measurements through a method shown in the ‘Using_the_Data’ notebook."
    # Since I don't have the notebook, I'll just plot raw flux for now, or use a simple correction if I find it.
    
    plt.figure(figsize=(12, 6))
    filters = lc['Filter'].unique()
    colors = {'u': 'purple', 'g': 'blue', 'r': 'green', 'i': 'orange', 'z': 'red', 'y': 'brown'}
    
    for f in sorted(filters):
        f_lc = lc[lc['Filter'] == f]
        plt.errorbar(f_lc['Time (MJD)'], f_lc['Flux'], yerr=f_lc['Flux_err'], fmt='o', label=f"Filter {f}", color=colors.get(f, None), alpha=0.7)
        
    plt.title(f"Lightcurve for {object_id} ({spec_type}, Target: {target}, EBV: {ebv})")
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (μJy)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot instead of showing it if in a non-interactive environment
    plt.savefig(f"lc_{object_id}.png")
    print(f"Plot saved as lc_{object_id}.png")
    # plt.show()

if __name__ == "__main__":
    # Plot a TDE and a non-TDE (AGN)
    plot_lightcurve("Dornhoth_fervain_onodrim") # AGN from previous peek
    # Let's find a TDE
    train_log = pd.read_csv("mallorn-astronomical-classification-challenge/train_log.csv")
    tde_id = train_log[train_log['target'] == 1].iloc[0]['object_id']
    plot_lightcurve(tde_id)
