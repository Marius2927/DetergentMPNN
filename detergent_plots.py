import matplotlib.pyplot as plt
import pandas as pd
import seaborn

folds = 5
epochs = 100
metrics = ['train_ddG_r2', 'train_ddG_spearman', 'train_ddG_mse', 'train_ddG_rmse', 'val_ddG_r2', 'val_ddG_spearman', 'val_ddG_mse', 'val_ddG_rmse']


df = pd.read_csv("data/metrics.csv", sep=',')
train_df = df[df['val_ddG_r2'].isna()].copy()  # Rows where validation metrics are NaN
val_df = df[df['train_ddG_r2'].isna()].copy()  # Rows where training metrics are NaN

# Add a fold number (assuming each fold has 100 epochs)
num_epochs_per_fold = 200  # Change this if needed
train_df['fold'] = train_df.index // num_epochs_per_fold
val_df['fold'] = val_df.index // num_epochs_per_fold

# Drop unnecessary NaN columns
train_df.drop(columns=['step', 'val_ddG_r2', 'val_ddG_mse', 'val_ddG_rmse', 'val_ddG_spearman'], inplace=True)
val_df.drop(columns=['step', 'train_ddG_r2', 'train_ddG_mse', 'train_ddG_rmse', 'train_ddG_spearman'], inplace=True)

df = pd.merge(train_df, val_df, on=['epoch', 'fold'])
avg_per_epoch = df.groupby('epoch')[metrics].agg(['mean', 'std'])

print(avg_per_epoch['train_ddG_mse']['mean'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax1.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_mse', 'mean')], label="Train MSE (Mean)")
ax1.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_mse', 'mean')], label="Validation MSE (Mean)")
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_mse', 'mean')] - avg_per_epoch[('val_ddG_mse', 'std')], 
                 avg_per_epoch[('val_ddG_mse', 'mean')] + avg_per_epoch[('val_ddG_mse', 'std')], 
                 alpha=0.3, label="Training Std Dev")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.legend()

ax2.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_rmse', 'mean')], label="Train RMSE (Mean)")
ax2.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_rmse', 'mean')], label="Validation RMSE (Mean)")
ax2.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_rmse', 'mean')] - avg_per_epoch[('val_ddG_rmse', 'std')],
                 avg_per_epoch[('val_ddG_rmse', 'mean')] + avg_per_epoch[('val_ddG_rmse', 'std')],
                 alpha=0.3, label="Validation Std Dev")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("RMSE")
ax2.legend()

fig.suptitle("Train vs Validation Loss Over Time")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax1.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_r2', 'mean')], label="Train R² (Mean)", linestyle='-', marker='o', color='blue', alpha=0.7)
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('train_ddG_r2', 'mean')] - avg_per_epoch[('train_ddG_r2', 'std')],
                 avg_per_epoch[('train_ddG_r2', 'mean')] + avg_per_epoch[('train_ddG_r2', 'std')],
                 alpha=0.2, color='blue', label="Training Std Dev")

# Validation R² plot with std deviation (Line plot)
ax1.plot(avg_per_epoch.index, avg_per_epoch[('val_ddG_r2', 'mean')], label="Validation R² (Mean)", linestyle='-', marker='s', color='orange', alpha=0.7)
ax1.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('val_ddG_r2', 'mean')] - avg_per_epoch[('val_ddG_r2', 'std')],
                 avg_per_epoch[('val_ddG_r2', 'mean')] + avg_per_epoch[('val_ddG_r2', 'std')],
                 alpha=0.2, color='orange', label="Validation Std Dev")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("R²")
ax1.legend()
ax1.grid(True)

# Train Spearman plot with std deviation (Line plot)
ax2.plot(avg_per_epoch.index, avg_per_epoch[('train_ddG_spearman', 'mean')], label="Train Spearman (Mean)", linestyle='-', marker='o', color='green', alpha=0.7)
ax2.fill_between(avg_per_epoch.index,
                 avg_per_epoch[('train_ddG_spearman', 'mean')] - avg_per_epoch[('train_ddG_spearman', 'std')],
                 avg_per_epoch[('train_ddG_spearman', 'mean')] + avg_per_epoch[('train_ddG_spearman', 'std')],
                 alpha=0.2, color='green', label="Training Std Dev")

# Validation Spearman plot with std deviation (Line plot)
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
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 5))

# Plotting individual folds for training R²
avg =[]
for fold in range(5):  # Assuming 5 folds
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['train_ddG_r2'].mean())
    axs[0,0].scatter(fold_data['epoch'], fold_data['train_ddG_r2'], label=f"Fold {fold}", alpha=0.6)

axs[0,0].set_xlabel("Epoch")
axs[0,0].set_ylabel("R²")
axs[0,0].set_title("Scatter Plot of Training R² Across Epochs (Each Fold)")
axs[0,0].legend()
axs[0,0].grid(True)

colors = ['blue', 'orange', 'green', 'red', 'purple']

# Create the bar plot with different colors for each bar
axs[0,1].bar(range(len(avg)), avg, color=colors, alpha=0.5)
# Adding title and labels
axs[0,1].set_title('Average Training R² Across Epochs (Each Fold)')
axs[0,1].set_xlabel('Fold')
axs[0,1].set_ylabel('R²')

avg =[]
for fold in range(5):  # Assuming 5 folds
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['train_ddG_spearman'].mean())
    axs[1,0].scatter(fold_data['epoch'], fold_data['train_ddG_spearman'], label=f"Fold {fold}", alpha=0.6)

axs[1,0].set_xlabel("Epoch")
axs[1,0].set_ylabel("Spearman")
axs[1,0].set_title("Scatter Plot of Training Spearman Across Epochs (Each Fold)")
axs[1,0].legend()
axs[1,0].grid(True)

# Create the bar plot with different colors for each bar
axs[1,1].bar(range(len(avg)), avg, color=colors, alpha=0.5)
# Adding title and labels
axs[1,1].set_title('Average Training Spearman Across Epochs (Each Fold)')
axs[1,1].set_xlabel('Fold')
axs[1,1].set_ylabel('Spearman')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 5))

# Plotting individual folds for training R²
avg =[]
for fold in range(5):  # Assuming 5 folds
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['val_ddG_r2'].mean())
    axs[0,0].scatter(fold_data['epoch'], fold_data['val_ddG_r2'], label=f"Fold {fold}", alpha=0.6)

axs[0,0].set_xlabel("Epoch")
axs[0,0].set_ylabel("R²")
axs[0,0].set_title("Scatter Plot of Validation R² Across Epochs (Each Fold)")
axs[0,0].legend()
axs[0,0].grid(True)

colors = ['blue', 'orange', 'green', 'red', 'purple']

# Create the bar plot with different colors for each bar
axs[0,1].bar(range(len(avg)), avg, color=colors, alpha=0.5)
# Adding title and labels
axs[0,1].set_title('Average Validation R² Across Epochs (Each Fold)')
axs[0,1].set_xlabel('Fold')
axs[0,1].set_ylabel('R²')

avg =[]
for fold in range(5):  # Assuming 5 folds
    fold_data = df[df['fold'] == fold]
    avg.append(fold_data['val_ddG_spearman'].mean())
    axs[1,0].scatter(fold_data['epoch'], fold_data['val_ddG_spearman'], label=f"Fold {fold}", alpha=0.6)

axs[1,0].set_xlabel("Epoch")
axs[1,0].set_ylabel("Spearman")
axs[1,0].set_title("Scatter Plot of Validation Spearman Across Epochs (Each Fold)")
axs[1,0].legend()
axs[1,0].grid(True)

# Create the bar plot with different colors for each bar
axs[1,1].bar(range(len(avg)), avg, color=colors, alpha=0.5)
# Adding title and labels
axs[1,1].set_title('Average Validation Spearman Across Epochs (Each Fold)')
axs[1,1].set_xlabel('Fold')
axs[1,1].set_ylabel('Spearman')
plt.show()

inference_df = pd.read_csv("ssm.csv", sep=',')  # Load your inference CSV
plt.hist(inference_df["ddG (kcal/mol)"], bins=30, alpha=0.75, color="blue")
plt.axvline(0, color='red', linestyle="--", label="Neutral Stability (ddG = 0)")
plt.xlabel("Predicted ddG (kcal/mol)")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted ddG Values")
plt.legend()
plt.show()

inference_df["Position"] = inference_df["Mutation"].str.extract(r'(\d+)').astype(int)

# Extract mutation type (e.g., GA141W -> 'A→W')
inference_df["Mutation Type"] = inference_df["Mutation"].str[0] + "→" + inference_df["Mutation"].str[-1]

heatmap_data = inference_df.groupby(["Position", "Mutation Type"])["ddG (kcal/mol)"].mean().reset_index()

# Now pivot without duplicates
heatmap_pivot = heatmap_data.pivot(index="Position", columns="Mutation Type", values="ddG (kcal/mol)")

# Create a heatmap
plt.figure(figsize=(12, 6))
seaborn.heatmap(heatmap_pivot, cmap="coolwarm", center=0, annot=False, linewidths=0.5)
plt.title("Mutation Effect Heatmap (Predicted ddG Values)")
plt.xlabel("Mutation Type")
plt.ylabel("Position")
plt.show()