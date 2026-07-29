import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def find_data_file(filename: str, start_dir: Path | None = None) -> Path:
    start = (start_dir or Path(__file__).resolve()).resolve()
    for candidate_dir in [start, *start.parents]:
        if (candidate_dir / filename).exists():
            return candidate_dir / filename
        if (candidate_dir / "Titanic" / filename).exists():
            return candidate_dir / "Titanic" / filename

    cwd_candidate = Path.cwd() / filename
    if cwd_candidate.exists():
        return cwd_candidate

    return start.parent.parent / filename


TRAIN_PATH = find_data_file("train.csv")

sns.set_theme(style="whitegrid", context="notebook")
SURVIVAL_ORDER = ["Did not survive", "Survived"]
SURVIVAL_PALETTE = {"Did not survive": "#d95f5f", "Survived": "#2f9e78"}
RATE_COLOR = "#4c78a8"


def load_train_data(path: Path = TRAIN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    if not match:
        return "Unknown"

    title = match.group(1).strip()
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Sir": "Royalty",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
    }
    return title_map.get(title, title)


def prepare_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SurvivalLabel"] = df["Survived"].map({0: "Did not survive", 1: "Survived"})
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].fillna("Unknown").astype(str).str[0].replace("U", "Unknown")
    return df


def summary_statistics(df: pd.DataFrame) -> None:
    print("\n== General information ==")
    print(df.info())
    print("\n== Numeric description ==")
    print(df.describe(include=["number"]))
    print("\n== Categorical description ==")
    print(df.describe(include=["object"]))


def missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isna().sum().sort_values(ascending=False)
    print("\n== Missing values ==")
    print(missing[missing > 0])
    return missing


def annotate_rate_bars(ax) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            ax.annotate(
                f"{height:.0%}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 4),
                textcoords="offset points",
            )


def plot_target_distribution(df: pd.DataFrame) -> None:
    survival_counts = (
        df["SurvivalLabel"]
        .value_counts(normalize=True)
        .reindex(SURVIVAL_ORDER)
        .rename("share")
        .reset_index()
    )
    survival_counts.columns = ["Survival", "Share"]

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        data=survival_counts,
        x="Survival",
        y="Share",
        hue="Survival",
        order=SURVIVAL_ORDER,
        palette=SURVIVAL_PALETTE,
        legend=False,
    )
    ax.set_title("Target distribution", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Passenger share")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    annotate_rate_bars(ax)
    plt.tight_layout()
    plt.show()


def plot_survival_rate(df: pd.DataFrame, feature: str, title: str, order=None, rotation: int = 0) -> None:
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(data=df, x=feature, y="Survived", order=order, errorbar=None, color=RATE_COLOR)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=rotation)
    annotate_rate_bars(ax)
    plt.tight_layout()
    plt.show()


def plot_key_survival_rates(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    specs = [
        ("Sex", "Survival by sex", None, 0),
        ("Pclass", "Survival by passenger class", [1, 2, 3], 0),
        ("Embarked", "Survival by embarkation port", None, 0),
        ("FamilySize", "Survival by family size", sorted(df["FamilySize"].unique()), 0),
    ]

    for ax, (feature, title, order, rotation) in zip(axes.flat, specs):
        sns.barplot(data=df, x=feature, y="Survived", order=order, errorbar=None, color=RATE_COLOR, ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Survival rate")
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=rotation)
        annotate_rate_bars(ax)

    plt.tight_layout()
    plt.show()


def plot_age_fare_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        data=df,
        x="Age",
        hue="SurvivalLabel",
        hue_order=SURVIVAL_ORDER,
        multiple="layer",
        stat="density",
        common_norm=False,
        bins=28,
        alpha=0.45,
        palette=SURVIVAL_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Age distribution by survival", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Density")
    axes[0].grid(axis="y", alpha=0.25)

    sns.boxplot(
        data=df,
        x="SurvivalLabel",
        y="Fare",
        hue="SurvivalLabel",
        order=SURVIVAL_ORDER,
        palette=SURVIVAL_PALETTE,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Fare distribution by survival", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Fare")
    axes[1].set_ylim(0, df["Fare"].quantile(0.98))
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_title_survival(df: pd.DataFrame) -> None:
    title_order = df.groupby("Title")["Survived"].mean().sort_values(ascending=False).index
    plot_survival_rate(df, "Title", "Survival by extracted title", order=title_order, rotation=35)


def plot_correlation(df: pd.DataFrame) -> None:
    numeric = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]]
    corr = numeric.corr()

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation matrix", fontweight="bold")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = prepare_eda_features(load_train_data())

    summary_statistics(df)
    missing_values(df)

    plot_target_distribution(df)
    plot_key_survival_rates(df)
    plot_age_fare_distribution(df)
    plot_title_survival(df)
    plot_correlation(df)


if __name__ == "__main__":
    main()
