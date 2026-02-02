import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_target_distribution(train: pd.DataFrame, save_dir: str = "plots/"):
    """
    Plots the distribution of the target variable 'isFraud' in the training dataset.
    
    Parameters:
    train (pd.DataFrame): The training dataset containing the 'isFraud' column.
    """
    print("Target Distribution:")
    print(train['isFraud'].value_counts())
    print(f"\nFraud Rate: {train['isFraud'].mean() * 100:.2f}%")

    # Visualize target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train['isFraud'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
    axes[0].set_title('Fraud vs Non-Fraud Count')
    axes[0].set_xlabel('isFraud')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Not Fraud (0)', 'Fraud (1)'], rotation=0)

    train['isFraud'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.2f%%', 
                                        colors=['steelblue', 'coral'], labels=['Not Fraud', 'Fraud'])
    axes[1].set_title('Fraud Distribution')
    axes[1].set_ylabel('')

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'target_distribution.png'))

    # Column categories
    print("\nColumn Categories:")
    print(f"- C columns (counting): {[c for c in train.columns if c.startswith('C')]}")
    print(f"- D columns (timedelta): {[c for c in train.columns if c.startswith('D')]}")
    print(f"- M columns (match): {[c for c in train.columns if c.startswith('M')]}")
    print(f"- V columns (Vesta features): {len([c for c in train.columns if c.startswith('V')])} columns")
    print(f"- id columns: {[c for c in train.columns if c.startswith('id_')]}")
