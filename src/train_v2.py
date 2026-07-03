"""
Improved training pipeline for Predicting Student Health Risk.

Key upgrades vs. the original notebook pipeline:
  1. No median imputation for XGBoost/LightGBM -- both handle NaN natively by
     learning an optimal split direction for missing values, so imputing
     medians was throwing away real signal (missingness itself correlates
     with the target here).
  2. Explicit missing-value indicator features (per-column + row total),
     since missingness patterns differ across classes.
  3. CatBoost added as a third, diverse model that consumes raw categorical
     columns natively (no one-hot / ordinal lossy encoding) plus native NaN
     handling for numeric columns.
  4. 5-fold CV instead of 2-fold for the final OOF/test averaging (was
     leaving each fold with only 50% train data).
  5. Early stopping (using each fold's own validation split) instead of a
     fixed n_estimators picked from a small 100k-row Optuna subsample.
  6. Weighted ensemble (weights grid-searched on OOF predictions to maximize
     accuracy) instead of a plain average across models.
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)

t_start = time.time()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_DIR = '../data'
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

TARGET = 'health_condition'
le = LabelEncoder()
y = le.fit_transform(train[TARGET])
n_classes = len(le.classes_)
print('Classes:', le.classes_)
print(f'Train shape: {train.shape}  Test shape: {test.shape}')

RAW_CAT_COLS = ['diet_type', 'stress_level', 'sleep_quality',
                'physical_activity_level', 'smoking_alcohol', 'gender']
RAW_NUM_COLS = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
                'step_count', 'exercise_duration', 'water_intake']
MISSING_SRC_COLS = RAW_CAT_COLS + RAW_NUM_COLS

stress_map = {'low': 0, 'medium': 1, 'high': 2}
quality_map = {'poor': 0, 'average': 1, 'good': 2}
activity_map = {'sedentary': 0, 'moderate': 1, 'active': 2}
smoke_map = {'no': 0, 'occasional': 1, 'yes': 2}


def add_missing_flags(df):
    df = df.copy()
    for col in MISSING_SRC_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(np.int8)
    df['total_missing'] = df[[f'{c}_missing' for c in MISSING_SRC_COLS]].sum(axis=1)
    return df


def add_interactions(df, stress_ord):
    df = df.copy()
    fill_stress = stress_ord.fillna(1)
    df['active_index'] = df['step_count'] / (df['exercise_duration'] + 1)
    df['calorie_per_step'] = df['calorie_expenditure'] / (df['step_count'] + 1)
    df['sleep_stress'] = df['sleep_duration'] * (3 - fill_stress)
    df['hydration_index'] = df['water_intake'] / (df['bmi'] + 1)
    df['health_score'] = (
        df['step_count'] / 10000
        + df['sleep_duration'] / 8
        + df['water_intake'] / 3
        - df['bmi'] / 30
        - fill_stress / 2
    )
    return df


def engineer_tree_features(df):
    """Ordinal + one-hot encoded feature set for XGBoost / LightGBM (NaN kept)."""
    df = add_missing_flags(df)

    df['stress_ord'] = df['stress_level'].map(stress_map)
    df['quality_ord'] = df['sleep_quality'].map(quality_map)
    df['activity_ord'] = df['physical_activity_level'].map(activity_map)
    df['smoke_ord'] = df['smoking_alcohol'].map(smoke_map)
    df['bmi_activity'] = df['bmi'] / (df['activity_ord'].fillna(1) + 1)

    df = add_interactions(df, df['stress_ord'])

    df = pd.get_dummies(df, columns=['diet_type', 'gender'], drop_first=False, dummy_na=True)

    df.drop(columns=['stress_level', 'sleep_quality',
                      'physical_activity_level', 'smoking_alcohol'], inplace=True)
    return df


def engineer_catboost_features(df):
    """Raw categorical columns (as strings) + numeric/interaction features for CatBoost."""
    df = add_missing_flags(df)

    # ordinal used only to build the same interaction terms as the tree-model set
    stress_ord = df['stress_level'].map(stress_map)
    activity_ord = df['physical_activity_level'].map(activity_map)
    df = add_interactions(df, stress_ord)
    df['bmi_activity'] = df['bmi'] / (activity_ord.fillna(1) + 1)

    for col in RAW_CAT_COLS:
        df[col] = df[col].fillna('missing').astype(str)

    return df


id_col = test['id'].copy()

feature_drop = ['id', TARGET]
train_feat = train.drop(columns=[TARGET])
test_feat = test.copy()

print('Building XGB/LGBM feature set...')
X_train_tree = engineer_tree_features(train_feat).drop(columns=['id'])
X_test_tree = engineer_tree_features(test_feat).drop(columns=['id'])
# align columns (dummy_na / rare categories could differ)
X_train_tree, X_test_tree = X_train_tree.align(X_test_tree, join='left', axis=1, fill_value=0)
print(f'Tree feature set: {X_train_tree.shape[1]} columns')

print('Building CatBoost feature set...')
X_train_cat = engineer_catboost_features(train_feat).drop(columns=['id'])
X_test_cat = engineer_catboost_features(test_feat).drop(columns=['id'])
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
cat_feature_idx = [X_train_cat.columns.get_loc(c) for c in RAW_CAT_COLS]
print(f'CatBoost feature set: {X_train_cat.shape[1]} columns, '
      f'{len(cat_feature_idx)} categorical')

# ---------------------------------------------------------------------------
# 2. Hyperparameters (base: Optuna-tuned values from the original notebook,
#    n_estimators raised + early stopping added so each fold picks its own
#    optimal iteration count instead of trusting a 100k-row subsample search).
# ---------------------------------------------------------------------------
lgbm_params = dict(
    n_estimators=2000, learning_rate=0.0562, num_leaves=35, max_depth=5,
    min_child_samples=39, subsample=0.8918, colsample_bytree=0.8550,
    reg_alpha=2.7294, reg_lambda=0.02297, random_state=SEED, n_jobs=-1, verbose=-1,
)
xgb_params = dict(
    n_estimators=2000, learning_rate=0.0521, max_depth=6, subsample=0.6558,
    colsample_bytree=0.7169, min_child_weight=8, gamma=2.2803,
    reg_alpha=0.8431, reg_lambda=0.000996, eval_metric='mlogloss',
    random_state=SEED, tree_method='hist', device='cpu', n_jobs=-1,
    early_stopping_rounds=50,
)
cat_params = dict(
    iterations=3000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    loss_function='MultiClass', random_seed=SEED, thread_count=-1,
    verbose=False, early_stopping_rounds=50, cat_features=cat_feature_idx,
    train_dir='../outputs/catboost_info',
)

N_FOLDS = 5
FINAL_CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = {name: np.zeros((len(train), n_classes)) for name in ['LightGBM', 'XGBoost', 'CatBoost']}
test_preds = {name: np.zeros((len(test), n_classes)) for name in ['LightGBM', 'XGBoost', 'CatBoost']}

for fold, (tr_idx, val_idx) in enumerate(FINAL_CV.split(X_train_tree, y)):
    print(f'\n===== FOLD {fold + 1}/{N_FOLDS} =====', flush=True)
    y_tr, y_val = y[tr_idx], y[val_idx]

    # ---- LightGBM ----
    t0 = time.time()
    Xtr, Xval = X_train_tree.iloc[tr_idx], X_train_tree.iloc[val_idx]
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(Xtr, y_tr, eval_set=[(Xval, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    val_prob = model.predict_proba(Xval)
    oof_preds['LightGBM'][val_idx] = val_prob
    test_preds['LightGBM'] += model.predict_proba(X_test_tree) / N_FOLDS
    acc = accuracy_score(y_val, val_prob.argmax(axis=1))
    print(f'  LightGBM   ACC={acc:.4f}  best_iter={model.best_iteration_}  ({time.time()-t0:.0f}s)', flush=True)

    # ---- XGBoost ----
    t0 = time.time()
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(Xtr, y_tr, eval_set=[(Xval, y_val)], verbose=False)
    val_prob = model.predict_proba(Xval)
    oof_preds['XGBoost'][val_idx] = val_prob
    test_preds['XGBoost'] += model.predict_proba(X_test_tree) / N_FOLDS
    acc = accuracy_score(y_val, val_prob.argmax(axis=1))
    print(f'  XGBoost    ACC={acc:.4f}  best_iter={model.best_iteration}  ({time.time()-t0:.0f}s)', flush=True)

    # ---- CatBoost ----
    t0 = time.time()
    Xtr_c, Xval_c = X_train_cat.iloc[tr_idx], X_train_cat.iloc[val_idx]
    model = CatBoostClassifier(**cat_params)
    model.fit(Xtr_c, y_tr, eval_set=(Xval_c, y_val), use_best_model=True)
    val_prob = model.predict_proba(Xval_c)
    oof_preds['CatBoost'][val_idx] = val_prob
    test_preds['CatBoost'] += model.predict_proba(X_test_cat) / N_FOLDS
    acc = accuracy_score(y_val, val_prob.argmax(axis=1))
    print(f'  CatBoost   ACC={acc:.4f}  best_iter={model.get_best_iteration()}  ({time.time()-t0:.0f}s)', flush=True)

# ---------------------------------------------------------------------------
# 3. Per-model OOF accuracy
# ---------------------------------------------------------------------------
print('\n===== OOF ACCURACY (5-fold) =====')
for name in oof_preds:
    acc = accuracy_score(y, oof_preds[name].argmax(axis=1))
    print(f'  {name:10s}: {acc:.4f}')

# ---------------------------------------------------------------------------
# 4. Weighted ensemble -- grid search weights on OOF to maximize accuracy
# ---------------------------------------------------------------------------
names = list(oof_preds.keys())
best_acc, best_w = -1, None
step = 0.05
grid = np.arange(0, 1.0001, step)
for w1 in grid:
    for w2 in grid:
        w3 = 1 - w1 - w2
        if w3 < -1e-9 or w3 > 1 + 1e-9:
            continue
        w3 = max(w3, 0)
        blend = w1 * oof_preds[names[0]] + w2 * oof_preds[names[1]] + w3 * oof_preds[names[2]]
        acc = accuracy_score(y, blend.argmax(axis=1))
        if acc > best_acc:
            best_acc, best_w = acc, (w1, w2, w3)

print(f'\nBest ensemble weights {dict(zip(names, best_w))}  -> OOF ACC = {best_acc:.4f}')

ensemble_oof = sum(w * oof_preds[n] for w, n in zip(best_w, names))
ensemble_test = sum(w * test_preds[n] for w, n in zip(best_w, names))

y_pred_oof = ensemble_oof.argmax(axis=1)
print('\n' + classification_report(y, y_pred_oof, target_names=le.classes_))
print(confusion_matrix(y, y_pred_oof))

# ---------------------------------------------------------------------------
# 5. Submission
# ---------------------------------------------------------------------------
final_preds = le.inverse_transform(ensemble_test.argmax(axis=1))
submission = pd.DataFrame({'id': id_col, TARGET: final_preds})
submission.to_csv('../outputs/submission.csv', index=False)
print(f'\nSaved submission.csv ({len(submission):,} rows)')
print(submission[TARGET].value_counts())

print(f'\nTotal runtime: {(time.time() - t_start) / 60:.1f} min')
