import os

import pandas as pd
import matplotlib.pyplot as plt

# Load your datasets
backbone_list = {}
pdb_prefix = 'NZPROT'
for file in os.listdir('data/nvs'):
    pdb_name = file[file.index(pdb_prefix):]
    pdb_name = pdb_name[:pdb_name.find('_', pdb_name.find('_') + 1)]
    detergent_df = pd.read_csv("data/nvs/"+file)  # adjust path
    detergent_df = detergent_df[(~detergent_df['mutant'].str.contains(',')) & detergent_df['mutant'].str.match(r'^[A-Z]\d+[A-Z]$')]
    mean_ddg = detergent_df['DMS_score'].mean()
    std_ddg = detergent_df['DMS_score'].std()
    lower_bound = mean_ddg - 3 * std_ddg
    upper_bound = mean_ddg + 3 * std_ddg

    detergent_df = detergent_df[(detergent_df['DMS_score'] >= lower_bound) & (detergent_df['DMS_score'] <= upper_bound)]
    detergent_df = detergent_df[(detergent_df['DMS_score'] >= -20) & (detergent_df['DMS_score'] <= 20)]
    #detergent_ddg = detergent_df['DMS_score']
    if pdb_name not in backbone_list.keys():
        backbone_list[pdb_name]=detergent_df
    else:
        backbone_list[pdb_name] = pd.concat([backbone_list[pdb_name], detergent_df], ignore_index=True)
print(len(backbone_list))
megascale_df = pd.read_csv("data/mega_train.csv")       # adjust path
# Extract ddG (or DMS) values
megascale_ddg = megascale_df['ddG_ML']
# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(megascale_ddg, bins=50, alpha=0.6, label='MegaScale ddG', color='orange')
plt.xlabel("ddG Value")
plt.ylabel("Frequency")
plt.title("Megascale Distributions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(12, 10))
axs = axs.flatten()  # Make it easier to index as a 1D array

for idx, (name, detergent_df) in enumerate(backbone_list.items()):
    axs[idx].hist(detergent_df['DMS_score'], bins=50, alpha=0.6, color='blue')
    axs[idx].set_title(name)
    axs[idx].set_xlabel('ddG Value')
    axs[idx].set_ylabel('Frequency')
    axs[idx].grid(True)

# Hide unused subplots if any
for j in range(len(backbone_list), 6):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()