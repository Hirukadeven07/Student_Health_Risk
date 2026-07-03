"""One-off script to generate README visualization assets from data/train.csv.
Run with: python scripts/make_readme_assets.py
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

OUT_DIR = 'assets'
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'health_condition'
CLASS_ORDER = ['fit', 'at-risk', 'unhealthy']
PALETTE = dict(zip(CLASS_ORDER, sns.color_palette('Set2', len(CLASS_ORDER))))

NUM_COLS = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
            'step_count', 'exercise_duration', 'water_intake']
CAT_COLS = ['diet_type', 'stress_level', 'sleep_quality',
            'physical_activity_level', 'smoking_alcohol', 'gender']

train = pd.read_csv('data/train.csv')
print('Loaded', train.shape)

# ---------------------------------------------------------------------------
# 1. Target class distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
vc = train[TARGET].value_counts().reindex(CLASS_ORDER)
colors = [PALETTE[c] for c in vc.index]
bars = ax.bar(vc.index, vc.values, color=colors, edgecolor='white', linewidth=1.2)
for i, (k, v) in enumerate(vc.items()):
    ax.text(i, v + len(train) * 0.01, f'{v:,}\n({v/len(train)*100:.1f}%)',
            ha='center', fontsize=10, fontweight='bold')
ax.set_title('Target Class Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('health_condition')
ax.set_ylabel('Count')
ax.set_ylim(0, vc.max() * 1.22)
sns.despine()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/target_distribution.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 2. Missing values by column
# ---------------------------------------------------------------------------
miss = train.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
pct = (miss / len(train) * 100).round(2)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(miss.index[::-1], pct.values[::-1],
                color=sns.color_palette('Blues_r', len(miss)))
for bar, v in zip(bars, pct.values[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('% Missing')
ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(pct.values) * 1.2)
sns.despine()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/missing_values.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Numeric features by target (density)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    for label in CLASS_ORDER:
        subset = train.loc[train[TARGET] == label, col].dropna()
        axes[i].hist(subset, bins=40, alpha=0.55, label=label, density=True, color=PALETTE[label])
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
axes[-1].set_visible(False)
fig.suptitle('Numeric Feature Distributions by Health Condition', fontsize=15, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/numeric_by_target.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------
corr = train[NUM_COLS].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Numeric Feature Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/correlation_heatmap.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Categorical features vs target
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(CAT_COLS):
    ct = pd.crosstab(train[col], train[TARGET], normalize='index')[CLASS_ORDER] * 100
    ct.plot(kind='bar', ax=axes[i], color=[PALETTE[c] for c in CLASS_ORDER],
            edgecolor='white', width=0.7, legend=(i == 0))
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('% within category')
    axes[i].tick_params(axis='x', rotation=25)
    if i == 0:
        axes[i].legend(fontsize=8, title=TARGET)
fig.suptitle('Categorical Features vs Health Condition', fontsize=15, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/categorical_vs_target.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Model comparison bar chart (from CV results observed in outputs/logs/train_v2_run.log)
# ---------------------------------------------------------------------------
model_scores = {
    'LightGBM': [0.9674, 0.9672, 0.9671, 0.9672, 0.9672],
    'XGBoost':  [0.9671, 0.9675, 0.9673, 0.9669, 0.9672],
    'CatBoost': [0.9666, 0.9673, 0.9666, 0.9670, 0.9670],
}
names = list(model_scores.keys())
means = [np.mean(v) for v in model_scores.values()]
stds = [np.std(v) for v in model_scores.values()]
ensemble_mean = 0.9683

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = sns.color_palette('Set2', len(names))
bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor='white', capsize=5)
for bar, m in zip(bars, means):
    ax.text(m + 0.0006, bar.get_y() + bar.get_height() / 2, f'{m:.4f}', va='center', fontsize=10)
ax.axvline(ensemble_mean, color='#444', linestyle='--', linewidth=1.5, label=f'Ensemble  {ensemble_mean:.4f}')
ax.set_xlabel('5-Fold CV Accuracy')
ax.set_title('Model Comparison — Cross-Validated Accuracy', fontsize=14, fontweight='bold')
ax.set_xlim(min(means) - 0.004, max(ensemble_mean, max(means)) + 0.004)
ax.legend(fontsize=9, loc='lower right')
sns.despine()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/model_comparison.png', dpi=150)
plt.close()

print('Saved assets to', OUT_DIR)
