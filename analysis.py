# wages_analysis_optimized.py
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

# CONFIG
RANDOM_STATE = 42
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Wages.csv"
APPLY_OUTLIER_REMOVAL = False   # toggle whether to remove lwage outliers before modeling

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data(url=DATA_URL):
    df = pd.read_csv(url)
    if 'rownames' in df.columns:
        df = df.drop(columns=['rownames'])
    return df


def safe_shapiro(data_series, sample_size=1000):
    n = len(data_series.dropna())
    if n < 3:
        return None, None
    sample_n = min(n, sample_size)
    sample = data_series.dropna().sample(sample_n, random_state=RANDOM_STATE)
    stat, p = stats.shapiro(sample)
    return stat, p


def detect_outliers_iqr(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lb) | (series > ub)
    return mask, lb, ub


def prepare_features(df, drop_target=True):
    """
    Return X (features) and y (target) and lists of numeric/categorical columns.
    """
    df = df.copy()
    target = 'lwage'
    if drop_target:
        y = df[target].copy()
        X = df.drop(columns=[target])
    else:
        X = df.copy()
        y = None

    # Identify numeric and categorical (conservative)
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    # If any numeric-looking columns are really categorical (like 'sex','union'), keep them in cat_cols
    # common columns in this dataset:
    for col in ['sex', 'union', 'married', 'black', 'south', 'smsa', 'bluecol']:
        if col in numeric_cols:
            numeric_cols.remove(col)
            cat_cols.append(col)

    return X, y, numeric_cols, cat_cols


def build_regression_pipelines(numeric_cols, cat_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop='first'))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder='drop')

    # Linear regression pipeline
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("model", LinearRegression())
    ])

    # Random forest pipeline (trees don't need scaling but we keep imputation/one-hot)
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))
    ])

    return pipe_lr, pipe_rf, preprocessor


def evaluate_model(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return dict(rmse=rmse, mse=mse, mae=mae, r2=r2, preds=preds)


def plot_actual_vs_pred(y_test, preds, title, savepath):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, preds, alpha=0.6)
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    plt.xlabel("Actual lwage")
    plt.ylabel("Predicted lwage")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def main():
    logging.info("Loading data...")
    df = load_data()
    logging.info(f"Rows: {len(df)}, Columns: {df.shape[1]}")

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=['lwage']).reset_index(drop=True)

    # ---------------------------
    # Q1: Simple descriptive checks
    # ---------------------------
    logging.info("\nQ1: Descriptive facts")
    # correlations (dropna)
    corr_ed = df['ed'].corr(df['lwage'])
    corr_exp = df['exp'].corr(df['lwage'])
    logging.info(f"Correlation ed vs lwage: {corr_ed:.3f}")
    logging.info(f"Correlation exp vs lwage: {corr_exp:.3f}")

    # Save a small figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby('ed')['lwage'].mean().plot.bar(ax=axes[0], rot=0)
    axes[0].set_title("Mean lwage by education (ed)")
    df.groupby('exp')['lwage'].mean().loc[:30].plot(ax=axes[1], marker='o')
    axes[1].set_title("Mean lwage by experience (exp) - first 30 years")
    df.groupby('sex')['lwage'].mean().plot.bar(ax=axes[2], rot=0)
    axes[2].set_title("Mean lwage by sex")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_summary.png", dpi=150)
    plt.close()

    # ---------------------------
    # Q2: Distribution
    # ---------------------------
    logging.info("\nQ2: Distribution")
    stat_sh, p_sh = safe_shapiro(df['lwage'], sample_size=500)
    logging.info(f"Shapiro sample p-value: {p_sh}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['lwage'], kde=True, ax=axes[0], bins=40)
    axes[0].axvline(df['lwage'].mean(), linestyle='--', label='mean')
    axes[0].axvline(df['lwage'].median(), linestyle='--', label='median')
    axes[0].legend()
    stats.probplot(df['lwage'].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q2_distribution.png", dpi=150)
    plt.close()

    # ---------------------------
    # Q3: Outlier detection
    # ---------------------------
    logging.info("\nQ3: Outliers (IQR)")
    mask_out, lb, ub = detect_outliers_iqr(df['lwage'])
    n_out = mask_out.sum()
    logging.info(f"IQR bounds: [{lb:.4f}, {ub:.4f}]  Outliers: {n_out} ({n_out/len(df):.1%})")

    if APPLY_OUTLIER_REMOVAL:
        df_modeling = df.loc[~mask_out].copy()
        logging.info(f"Applied outlier removal: new n = {len(df_modeling)}")
    else:
        df_modeling = df.copy()
        logging.info("Not removing outliers for modeling (configurable).")

    # ---------------------------
    # Q4: Assumption: union wage
    # ---------------------------
    logging.info("\nQ4: Union vs Non-union (Welch t-test)")
    union = df_modeling.loc[df_modeling['union']=='yes', 'lwage'].dropna()
    non_union = df_modeling.loc[df_modeling['union']=='no', 'lwage'].dropna()
    if len(union) >= 3 and len(non_union) >= 3:
        tstat, pval = stats.ttest_ind(union, non_union, equal_var=False)
        logging.info(f"union mean: {union.mean():.4f}, non-union mean: {non_union.mean():.4f}")
        logging.info(f"t-stat: {tstat:.4f}, p-val: {pval:.6f}")
    else:
        logging.info("Not enough data for t-test.")

    # ---------------------------
    # Q5: Modeling
    # ---------------------------
    logging.info("\nQ5: Modeling (LinearRegression vs RandomForest)")

    X, y, numeric_cols, cat_cols = prepare_features(df_modeling, drop_target=True)
    logging.info(f"Numeric cols: {numeric_cols}")
    logging.info(f"Categorical cols: {cat_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe_lr, pipe_rf, preprocessor = build_regression_pipelines(numeric_cols, cat_cols)

    # Evaluate with cross-validation (cv=5)
    logging.info("Cross-validating RandomForest (may take a moment)...")
    cv_scores = cross_val_score(pipe_rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    logging.info(f"RF CV RMSE (5-fold): {(-cv_scores).mean():.4f}")

    # Fit & evaluate
    results_lr = evaluate_model(pipe_lr, X_train, y_train, X_test, y_test)
    results_rf = evaluate_model(pipe_rf, X_train, y_train, X_test, y_test)

    logging.info(f"LR  => RMSE: {results_lr['rmse']:.4f}, MAE: {results_lr['mae']:.4f}, R2: {results_lr['r2']:.4f}")
    logging.info(f"RF  => RMSE: {results_rf['rmse']:.4f}, MAE: {results_rf['mae']:.4f}, R2: {results_rf['r2']:.4f}")

    plot_actual_vs_pred(y_test, results_lr['preds'], "LinearRegression: Actual vs Predicted", FIG_DIR / "q5_lr_actual_pred.png")
    plot_actual_vs_pred(y_test, results_rf['preds'], "RandomForest: Actual vs Predicted", FIG_DIR / "q5_rf_actual_pred.png")

    # Feature importance (RF - permutation importance after preprocessor)
    logging.info("Computing permutation importances for RF...")
    rf_pipe = pipe_rf
    rf_pipe.fit(X_train, y_train)
    # Transform test set to preprocessed features to use permutation_importance easily
    X_test_trans = rf_pipe.named_steps['pre'].transform(X_test)
    pi = permutation_importance(rf_pipe.named_steps['model'], X_test_trans, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    importances = pi.importances_mean
    # Recover feature names after one-hot
    num_features = numeric_cols
    cat_ohe = rf_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
    cat_names = []
    try:
        cat_names = cat_ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        # fallback
        cat_names = [f"{c}_{i}" for c in cat_cols for i in range(3)]
    feature_names = num_features + cat_names
    fi_df = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    logging.info("Top RF features (by permutation importance):")
    logging.info("\n" + fi_df.to_string())

    # ---------------------------
    # Q6: Clustering (KMeans)
    # ---------------------------
    logging.info("\nQ6: Clustering")
    cl_features = ['exp', 'ed', 'wks']
    cl_df = df_modeling[cl_features].dropna()
    scaler = StandardScaler()
    Xc = scaler.fit_transform(cl_df)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(Xc)
    cl_df = cl_df.assign(cluster=labels)
    cluster_means = cl_df.groupby('cluster').mean()
    logging.info("Cluster means:\n" + cluster_means.to_string())

    # Plot clusters (experience vs education)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=cl_df['exp'], y=cl_df['ed'], hue=cl_df['cluster'], palette='tab10', alpha=0.6)
    plt.title("KMeans clusters (exp vs ed)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q6_clusters.png", dpi=150)
    plt.close()

    # ---------------------------
    # Save results & data
    # ---------------------------
    df_modeling.to_csv("wages_analysis_cleaned.csv", index=False)
    logging.info("\nSaved cleaned dataset to wages_analysis_cleaned.csv")
    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()