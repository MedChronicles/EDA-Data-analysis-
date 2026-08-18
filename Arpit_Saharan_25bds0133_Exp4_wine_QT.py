import os
from dataclasses import dataclass
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set(style="ticks", color_codes=True)

DATA_PATH = "WineQT.csv"
PLOTS_DIR = "plots"


@dataclass
class DispersionSummary:
    column: str
    mean: float
    median: float
    mode: float
    std: float
    variance: float
    q1: float
    q3: float
    iqr: float


def ensure_plots_dir(path: str = PLOTS_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("First 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nDescriptive statistics:")
    print(df.describe())


def central_tendency_and_dispersion(df: pd.DataFrame, column: str) -> DispersionSummary:
    series = df[column]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    summary = DispersionSummary(
        column=column,
        mean=series.mean(),
        median=series.median(),
        mode=series.mode().iloc[0],
        std=series.std(),
        variance=series.var(),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
    )
    print(f"\nCentral tendency & dispersion for '{column}':")
    for field, value in summary.__dict__.items():
        if field != "column":
            print(f"  {field:>10}: {value:.4f}")
    return summary


def plot_distribution(df: pd.DataFrame, column: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_quality_counts(df: pd.DataFrame, filename: str = "quality_counts.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    df["quality"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Number of Wine Samples by Quality Score")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, column: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[column], ax=ax, color="steelblue")
    ax.set_title(f"Boxplot of {column}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def univariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)
    for col in ["alcohol", "pH"]:
        central_tendency_and_dispersion(df, col)
        plot_distribution(df, col, f"dist_{col}.png")
    plot_quality_counts(df)
    plot_boxplot(df, "alcohol", "box_alcohol.png")


def plot_scatter(df: pd.DataFrame, x: str, y: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x], df[y], alpha=0.6, color="mediumvioletred")
    ax.set_title(f"Scatter Plot: {x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_grouped_boxplot(df: pd.DataFrame, group_col: str, value_col: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=group_col, y=value_col, data=df, ax=ax)
    ax.set_title(f"{value_col} distribution across {group_col}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def bivariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("BIVARIATE ANALYSIS")
    print("=" * 60)
    plot_scatter(df, "alcohol", "quality", "scatter_alcohol_quality.png")
    plot_scatter(df, "volatile acidity", "quality", "scatter_volacid_quality.png")
    plot_grouped_boxplot(df, "quality", "alcohol", "box_alcohol_by_quality.png")
    plot_grouped_boxplot(df, "quality", "volatile acidity", "box_volacid_by_quality.png")


def plot_pairplot(df: pd.DataFrame, columns: Sequence[str], filename: str, hue: str = None) -> None:
    grid = sns.pairplot(df, vars=list(columns), hue=hue, kind="reg" if hue is None else "scatter", height=2.2)
    grid.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close("all")


def correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> pd.DataFrame:
    correlation = df.corr(method="pearson", numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns,
                cmap="rocket", annot=False, ax=ax)
    ax.set_title("Pearson Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)
    return correlation


def pearson_test(df: pd.DataFrame, col_a: str, col_b: str) -> None:
    corr, p_value = stats.pearsonr(df[col_a], df[col_b])
    print(f"\nPearson correlation between '{col_a}' and '{col_b}':")
    print(f"  correlation coefficient: {corr:.6f}")
    print(f"  p-value:                 {p_value:.6e}")


def multivariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 60)
    plot_pairplot(df, ["alcohol", "volatile acidity", "quality"], "pairplot_reg.png")
    plot_pairplot(df, ["citric acid", "sulphates", "alcohol"], "pairplot_quality_hue.png", hue="quality")
    correlation = correlation_heatmap(df)
    pearson_test(df, "alcohol", "quality")
    pearson_test(df, "volatile acidity", "quality")
    top_correlated_with_quality = (
        correlation["quality"].drop("quality").sort_values(key=np.abs, ascending=False)
    )
    print("\nFeatures most correlated with wine quality:")
    print(top_correlated_with_quality)


def main() -> None:
    ensure_plots_dir()
    df = load_dataset()
    inspect_dataset(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)
    print("\nAll plots saved to the 'plots' directory.")


if __name__ == "__main__":
    main()
