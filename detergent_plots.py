from traceback import print_tb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import os
import glob
from sklearn.linear_model import LinearRegression
from pymol import cmd

type = 'site'
thermo_performance = pd.read_csv("data/thermompnn_"+type+"_fold_performance.csv")
detergent_performance = pd.read_csv("data/detergent_"+type+"_fold_performance.csv")

reg_thermo = LinearRegression().fit(thermo_performance['preds'].values.reshape(-1, 1), thermo_performance['targets'].values.reshape(-1, 1))
reg_detergent = LinearRegression().fit(detergent_performance['preds'].values.reshape(-1, 1), detergent_performance['targets'].values.reshape(-1, 1))


y_pred_thermo = reg_thermo.predict(thermo_performance['preds'].values.reshape(-1, 1))
y_pred_detergent = reg_detergent.predict(detergent_performance['preds'].values.reshape(-1, 1))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(thermo_performance['preds'], thermo_performance['targets'], alpha=0.5)
ax1.plot(thermo_performance['preds'], y_pred_thermo, 'r-', label=f'Regression line (slope = {reg_thermo.coef_[0][0]:.2f})')
ax1.set_xlabel("Actual ddG")
ax1.set_ylabel("Predicted ddG")
ax1.legend()
ax1.set_title("ThermoMPNN Performance per " +type.capitalize())
ax1.grid(True)

ax2.scatter(detergent_performance['preds'], detergent_performance['targets'], alpha=0.5)
ax2.plot(detergent_performance['preds'], y_pred_detergent, 'r-', label=f'Regression line (slope = {reg_detergent.coef_[0][0]:.2f})')
ax2.set_xlabel("Actual ddG")
ax2.set_ylabel("Predicted ddG")
ax2.legend()
ax2.set_title("DetergentMPNN Performance per "+type.capitalize())
ax2.grid(True)

fig.suptitle("ThermoMPNN vs DetergentMPNN on detergent data")

plt.savefig('plots/'+type+'_fold/'+type+'_plt0.png')

metrics = ['train_ddG_r2', 'train_ddG_spearman', 'train_ddG_mse', 'train_ddG_rmse', 'val_ddG_r2', 'val_ddG_spearman', 'val_ddG_mse', 'val_ddG_rmse']

df = pd.DataFrame()
count = 1
for file in os.listdir('logs/'+type+'_fold'):
    if file.startswith("training"):
        list_of_files = glob.glob('logs/'+type+'_fold/' + file + '/*/metrics.csv')
        latest_file = max(list_of_files, key=os.path.getctime)
        fold_df = pd.read_csv(latest_file, sep=',')
        train_df = fold_df[fold_df['val_ddG_r2'].isna()].copy()
        val_df = fold_df[fold_df['train_ddG_r2'].isna()].copy()
        train_df.drop(columns=['step', 'val_ddG_r2', 'val_ddG_mse', 'val_ddG_rmse', 'val_ddG_spearman'], inplace=True)
        val_df.drop(columns=['step', 'train_ddG_r2', 'train_ddG_mse', 'train_ddG_rmse', 'train_ddG_spearman'],
                    inplace=True)
        train_df['fold'] = count
        val_df['fold'] = count
        fold_df = pd.merge(train_df, val_df, on=['epoch', 'fold'])
        df = pd.concat([df, fold_df], ignore_index=True)
        count+=1
avg_per_epoch = df.groupby('epoch')[metrics].agg(['mean', 'std'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_mse', 'mean')], label="Train MSE (Mean)")
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('train_ddG_mse', 'mean')] - avg_per_epoch[('train_ddG_mse', 'std')],
                 avg_per_epoch[('train_ddG_mse', 'mean')] + avg_per_epoch[('train_ddG_mse', 'std')],
                 alpha=0.3, color='blue', label="Training Std Dev")
ax1.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_mse', 'mean')], label="Validation MSE (Mean)")
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_mse', 'mean')] - avg_per_epoch[('val_ddG_mse', 'std')],
                 avg_per_epoch[('val_ddG_mse', 'mean')] + avg_per_epoch[('val_ddG_mse', 'std')],
                 alpha=0.3, color='orange', label="Validation Std Dev")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.legend()

ax2.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_rmse', 'mean')], label="Train RMSE (Mean)")
ax2.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('train_ddG_rmse', 'mean')] - avg_per_epoch[('train_ddG_rmse', 'std')],
                 avg_per_epoch[('train_ddG_rmse', 'mean')] + avg_per_epoch[('train_ddG_rmse', 'std')],
                 alpha=0.3, color='blue', label="Training Std Dev")
ax2.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_rmse', 'mean')], label="Validation RMSE (Mean)")
ax2.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_rmse', 'mean')] - avg_per_epoch[('val_ddG_rmse', 'std')],
                 avg_per_epoch[('val_ddG_rmse', 'mean')] + avg_per_epoch[('val_ddG_rmse', 'std')],
                 alpha=0.3, color='orange', label="Validation Std Dev")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("RMSE")
ax2.legend()

fig.suptitle("Train vs Validation Loss Over Time")
plt.savefig('plots/'+type+'_fold/'+type+'_plt1.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_r2', 'mean')], label="Train R² (Mean)", linestyle='-', marker='o', color='blue', alpha=0.7)
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('train_ddG_r2', 'mean')] - avg_per_epoch[('train_ddG_r2', 'std')],
                 avg_per_epoch[('train_ddG_r2', 'mean')] + avg_per_epoch[('train_ddG_r2', 'std')],
                 alpha=0.2, color='blue', label="Training Std Dev")

ax1.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_r2', 'mean')], label="Validation R² (Mean)", linestyle='-', marker='s', color='orange', alpha=0.7)
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_r2', 'mean')] - avg_per_epoch[('val_ddG_r2', 'std')],
                 avg_per_epoch[('val_ddG_r2', 'mean')] + avg_per_epoch[('val_ddG_r2', 'std')],
                 alpha=0.2, color='orange', label="Validation Std Dev")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("R²")
ax1.legend()
ax1.grid(True)

ax2.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_spearman', 'mean')], label="Train Spearman (Mean)", linestyle='-', marker='o', color='green', alpha=0.7)
ax2.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('train_ddG_spearman', 'mean')] - avg_per_epoch[('train_ddG_spearman', 'std')],
                 avg_per_epoch[('train_ddG_spearman', 'mean')] + avg_per_epoch[('train_ddG_spearman', 'std')],
                 alpha=0.2, color='green', label="Training Std Dev")

ax2.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_spearman', 'mean')], label="Validation Spearman (Mean)", linestyle='-', marker='s', color='red', alpha=0.7)
ax2.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_spearman', 'mean')] - avg_per_epoch[('val_ddG_spearman', 'std')],
                 avg_per_epoch[('val_ddG_spearman', 'mean')] + avg_per_epoch[('val_ddG_spearman', 'std')],
                 alpha=0.2, color='red', label="Validation Std Dev")

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Spearman Correlation")
ax2.legend()
ax2.grid(True)

fig.suptitle("Training and Validation Spearman Correlation (Mean ± Std) Over Epochs")
plt.savefig('plots/'+type+'_fold/'+type+'_plt2.png')

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

avg =[]
for fold in range(1,count):
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['train_ddG_r2'].mean())
    axs[0,0].scatter(fold_data['epoch'], fold_data['train_ddG_r2'], alpha=0.5)

axs[0,0].set_xlabel("Epoch")
axs[0,0].set_ylabel("R²")
axs[0,0].set_title("Scatter Plot of Training R² Across Epochs (Each Fold)")
axs[0,0].grid(True)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

axs[0,1].bar(range(1, count), avg, color=colors, alpha=0.5)
axs[0,1].set_title('Average Training R² Across Epochs (Each Fold)')
axs[0,1].set_xlabel('Fold')
axs[0,1].set_ylabel('R²')

avg =[]
for fold in range(1,count):
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['train_ddG_spearman'].mean())
    axs[1,0].scatter(fold_data['epoch'], fold_data['train_ddG_spearman'], alpha=0.5)

axs[1,0].set_xlabel("Epoch")
axs[1,0].set_ylabel("Spearman")
axs[1,0].set_title("Scatter Plot of Training Spearman Across Epochs (Each Fold)")
axs[1,0].grid(True)

axs[1,1].bar(range(1, count), avg, color=colors, alpha=0.5)
axs[1,1].set_title('Average Training Spearman Across Epochs (Each Fold)')
axs[1,1].set_xlabel('Fold')
axs[1,1].set_ylabel('Spearman')
if type == 'backbone' or type == 'all_backbone':
    pdb_ids = pd.read_csv('data/pdb_ids.csv')
    labels = pdb_ids['id'].values.tolist()
else:
    labels = ['Fold ' + str(i) for i in range(1, count)]
fig.legend(labels, loc='lower right', fontsize="x-large")
plt.savefig('plots/'+type+'_fold/'+type+'_plt3.png')

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

avg =[]
for fold in range(1,count):
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['val_ddG_r2'].mean())
    axs[0,0].scatter(fold_data['epoch'], fold_data['val_ddG_r2'], label=f"Fold {fold}", alpha=0.6)

axs[0,0].set_xlabel("Epoch")
axs[0,0].set_ylabel("R²")
axs[0,0].set_title("Scatter Plot of Validation R² Across Epochs (Each Fold)")
axs[0,0].grid(True)

axs[0,1].bar(range(1, count), avg, color=colors, alpha=0.5)
axs[0,1].set_title('Average Validation R² Across Epochs (Each Fold)')
axs[0,1].set_xlabel('Fold')
axs[0,1].set_ylabel('R²')

avg =[]
for fold in range(1,count):
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['val_ddG_spearman'].mean())
    axs[1,0].scatter(fold_data['epoch'], fold_data['val_ddG_spearman'], label=f"Fold {fold}", alpha=0.6)

axs[1,0].set_xlabel("Epoch")
axs[1,0].set_ylabel("Spearman")
axs[1,0].set_title("Scatter Plot of Validation Spearman Across Epochs (Each Fold)")
axs[1,0].grid(True)

axs[1,1].bar(range(1, count), avg, color=colors, alpha=0.5)
# Adding title and labels
axs[1,1].set_title('Average Validation Spearman Across Epochs (Each Fold)')
axs[1,1].set_xlabel('Fold')
axs[1,1].set_ylabel('Spearman')
fig.legend(labels, loc='lower right', fontsize="x-large")
plt.savefig('plots/'+type+'_fold/'+type+'_plt4.png')

plt.clf()
plt.figure(figsize=(10, 5))
thermo_inference_df = pd.read_csv("ssm_thermo.csv", sep=',')
inference_df = pd.read_csv("ssm_"+type+".csv", sep=',')
plt.hist(inference_df["ddG (kcal/mol)"], bins=30, alpha=0.75, color="blue")
plt.axvline(0, color='red', linestyle="--", label="Neutral Stability (ddG = 0)")
plt.xlabel("Predicted ddG (kcal/mol)")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted ddG Values")
plt.legend()
plt.savefig('plots/'+type+'_fold/'+type+'_plt5.png')

inference_df["Position"] = inference_df["Mutation"].str.extract(r'(\d+)').astype(int)
thermo_inference_df["Position"] = thermo_inference_df["Mutation"].str.extract(r'(\d+)').astype(int)

inference_df["Mutant AA"] = inference_df["Mutation"].str[-1]
thermo_inference_df["Mutant AA"] = thermo_inference_df["Mutation"].str[-1]

inference_df["WT AA"] = inference_df["Mutation"].str[0]
thermo_inference_df["WT AA"] = thermo_inference_df["Mutation"].str[0]

inference_df = inference_df[inference_df["WT AA"] != inference_df["Mutant AA"]]
thermo_inference_df = thermo_inference_df[thermo_inference_df["WT AA"] != thermo_inference_df["Mutant AA"]]

detergent_heatmap = inference_df.pivot_table(
    index="Mutant AA", columns="Position", values="ddG (kcal/mol)", aggfunc="mean"
)

thermo_heatmap = thermo_inference_df.pivot_table(
    index="Mutant AA", columns="Position", values="ddG (kcal/mol)", aggfunc="mean"
)

aa_order = list("ACDEFGHIKLMNPQRSTVWY")
detergent_heatmap = detergent_heatmap.reindex(aa_order)
thermo_heatmap = thermo_heatmap.reindex(aa_order)

plt.figure(figsize=(15, 5))
seaborn.heatmap(detergent_heatmap, cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'ddG (kcal/mol)'})
plt.title("Mutation Heatmap")
plt.xlabel("Position")
plt.ylabel("Amino Acid")
plt.tight_layout()
plt.savefig('plots/'+type+'_fold/'+type+'_plt8.png')

plt.figure(figsize=(15, 5))
seaborn.heatmap(thermo_heatmap, cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'ddG (kcal/mol)'})
plt.title("Mutation Heatmap")
plt.xlabel("Position")
plt.ylabel("Amino Acid")
plt.tight_layout()
plt.savefig('plots/thermompnn_heatmap.png')

delta_heatmap = detergent_heatmap - thermo_heatmap

plt.figure(figsize=(15, 5))
seaborn.heatmap(delta_heatmap, cmap="bwr", center=0, linewidths=0.5,
            cbar_kws={'label': 'ΔddG (Detergent - Thermo)'})
plt.title("Difference in Mutation Effects (DetergentMPNN - ThermoMPNN)")
plt.xlabel("Position")
plt.ylabel("Mutant Amino Acid")
plt.tight_layout()
plt.savefig("plots/"+type+"_fold/"+type+"_delta_heatmap.png")

stabilizing = inference_df[inference_df["ddG (kcal/mol)"] <= -0.5]
top_positions = stabilizing["Position"].unique()
print(stabilizing['Mutation'].values)

stabilizing['TargetAA'] = stabilizing['Mutation'].str[-1]
stabilizing['Position'] = stabilizing['Mutation'].str[2:-1].astype(int)
mean_ddg_AA = stabilizing.groupby('TargetAA')['ddG (kcal/mol)'].mean().reindex(aa_order).dropna(how='all')
mean_ddg_pos = stabilizing.sort_values('Position').groupby('Position')['ddG (kcal/mol)'].mean()

plt.figure(figsize=(10,5))
mean_ddg_AA.plot(kind='bar', edgecolor='black')
plt.xlabel('Mutated‐to Amino Acid')
plt.ylabel('Mean ddG (kcal/mol)')
plt.title('Average Predicted ddG by Target Amino Acid')
plt.axhline(0, color='gray', linewidth=0.8)
plt.tight_layout()
plt.savefig('plots/'+type+'_fold/'+type+'_plt6.png')

plt.figure(figsize=(10,5))
mean_ddg_pos.plot(kind='bar', edgecolor='black')
plt.xlabel('Mutated Position')
plt.ylabel('Mean ddG (kcal/mol)')
plt.title('Average Predicted ddG by Position')
plt.axhline(0, color='gray', linewidth=0.8)
plt.tight_layout()
plt.savefig('plots/'+type+'_fold/'+type+'_plt7.png')

cmd.load('data/pdbs/NZPROT_P6345Z.pdb', 'protein')
cmd.bg_color('black')
cmd.color('gray', 'protein')
pos_str = '+'.join(str(p) for p in top_positions)
sel_str = f'protein and resi {pos_str}'
cmd.select('mutations', sel_str)
cmd.show('sticks', 'mutations')
cmd.color('marine', 'mutations')
cmd.zoom('mutations', 5)
cmd.png('highlighted_mutations.png', width=1200, height=800, dpi=300, ray=1)
cmd.save('highlighted.pse')
#cmd.save('data/highlighted.pdb', 'protein')