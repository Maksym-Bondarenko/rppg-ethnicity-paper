from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

####################################################################################################
# DATA
####################################################################################################

# Externalized data for easier editing
DATA_ETHNICITY_DISTRIBUTION = {
    "Ethnicity": ["European (white)", "Asian (white)", "Skin tones (Fitzpatrick scale)",
                  "Caucasian, Asian", "Black, White, Asian, Latino, Native American", "N/A"],
    "Values": [141, 144, 142, 118, 80, 166]
}

DATA_PAPERS_PER_DATASET = {
    "Datasets": [
        "UBFC-rPPG", "PURE", "VIPL-HR", "COHFACE", "MMSE-HR", "MAHNOB-HCI",
        "UBFC-Phys", "MR-Nirp", "OBF", "BH-rPPG", "BUAA", "MPSC-rPPG",
        "ECG Fitness", "V4V", "TokyoTech", "MERL", "PFF", "LGI-PPGI",
        "DDPM", "UCLA-rPPG", "DEAP", "VicarPPG", "BP4D+", "CCUHR"
    ],
    "Papers": [52, 34, 27, 22, 14, 10, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    "MultiEthnicity": ["MMSE-HR", "BP4D+", "V4V", "MERL", "UCLA-rPPG", "MR-Nirp"]
}

DATA_GENDER_DISTRIBUTION = {
    "Datasets": [
        "UBFC", "PURE", "COHFACE", "MMSE-HR", "VIPL-HR", "LGI-PPG", "MR-NIRP (auto)",
        "MR-NIRP (indoor)", "OBF", "UBFC-Phys", "BP4D+", "TokyoTech", "MPSC-rPPG",
        "BH-rPPG", "BUAA-MIHR", "ECG-Fitness", "MERL", "DEAP"
    ],
    "Female": [11, 2, 12, 23, 28, 5, 2, 2, 39, 46, 82, 1, 2, 1, 3, 3, 3, 16],
    "Male": [31, 8, 28, 17, 79, 20, 16, 6, 61, 10, 58, 8, 6, 11, 12, 14, 9, 16]
}

DATA_ETHNICITY_BOXPLOT = {
    "Ethnicity": ["Asian", "Asian", "Asian", "Black", "Black", "Black",
                  "Other", "Other", "Other", "White", "White", "White"],
    "Proportion (%)": [10, 15, 12, 5, 7, 6, 20, 18, 22, 80, 85, 78]
}

####################################################################################################
# PLOT-FUNCTIONS
####################################################################################################

# Function: Ethnicity distribution (donut chart)
def plot_ethnicity_distribution():
    ethnicities = DATA_ETHNICITY_DISTRIBUTION["Ethnicity"]
    values = DATA_ETHNICITY_DISTRIBUTION["Values"]
    colors = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499"]

    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=ethnicities,
        autopct=lambda p: f"{p:.0f}%",
        startangle=140,
        colors=colors,
        pctdistance=0.85
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    ax.set_title("Ethnicity Distribution Across Datasets", fontsize=18, fontweight='bold')
    plt.setp(texts, size=12)
    plt.tight_layout()
    plt.show()

# Function: Papers per dataset (bar chart)
def plot_papers_per_dataset():
    datasets = DATA_PAPERS_PER_DATASET["Datasets"]
    papers = DATA_PAPERS_PER_DATASET["Papers"]
    multi_ethnicity = DATA_PAPERS_PER_DATASET["MultiEthnicity"]

    colors = ["#44AA99" if dataset in multi_ethnicity else "#CC6677" for dataset in datasets]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(datasets))
    bars = ax.bar(x, papers, color=colors, width=0.8)

    for bar, paper_count in zip(bars, papers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(paper_count),
                ha='center', va='bottom', fontsize=10)

    ax.set_title("Number of Papers Using Each Dataset", fontsize=18, fontweight="bold")
    ax.set_ylabel("Number of Papers", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#44AA99", lw=4, label="Multi-Ethnicity"),
            plt.Line2D([0], [0], color="#CC6677", lw=4, label="Predominantly Single Ethnicity")
        ],
        fontsize=12,
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()

def plot_gender_distribution_horizontal():
    datasets = DATA_GENDER_DISTRIBUTION["Datasets"]
    female = np.array(DATA_GENDER_DISTRIBUTION["Female"])
    male = np.array(DATA_GENDER_DISTRIBUTION["Male"])
    total = female + male

    # Sort datasets by total subjects (ascending order)
    sorted_indices = np.argsort(-total)  # Negative sign to sort in descending order
    datasets = np.array(datasets)[sorted_indices]
    female = female[sorted_indices]
    male = male[sorted_indices]
    total = total[sorted_indices]

    # Calculate percentages
    female_pct = (female / total * 100).astype(int)
    male_pct = (male / total * 100).astype(int)

    # Define top-used datasets to bold
    top_used_datasets = ["UBFC", "PURE", "VIPL-HR", "COHFACE", "MMSE-HR", "MAHNOB-HCI"]

    # Bold text styling for top-used datasets
    styled_datasets = [
        f"$\\bf{{{dataset}}}$" if dataset in top_used_datasets else dataset for dataset in datasets
    ]

    # Colors
    colors = {"Female": "#88CCEE", "Male": "#CC6677"}

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(datasets))

    ax.barh(y, female, color=colors["Female"], label='Female')
    ax.barh(y, male, left=female, color=colors["Male"], label='Male')

    # Adding percentages to bars
    for i in range(len(datasets)):
        ax.text(female[i] / 2, i, f"{female_pct[i]}%", va='center', ha='center', fontsize=9, color="white",
                weight="bold")
        ax.text(female[i] + male[i] / 2, i, f"{male_pct[i]}%", va='center', ha='center', fontsize=9, color="white",
                weight="bold")

    # Customization
    ax.set_yticks(y)
    ax.set_yticklabels(styled_datasets, fontsize=12)
    ax.set_title("Gender Distribution Across Datasets (Sorted by Total Subjects)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Number of Subjects", fontsize=14)
    ax.set_ylabel("Dataset", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_dataset_usage_distribution():
    # Data for dataset usage
    datasets = DATA_PAPERS_PER_DATASET["Datasets"]
    papers = DATA_PAPERS_PER_DATASET["Papers"]

    # Colors for the pie chart
    colors = sns.color_palette("pastel", len(datasets))

    # Plotting the donut chart
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        papers,
        labels=datasets,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=140,
        colors=colors,
        pctdistance=0.85
    )

    # Create a donut by adding a circle in the center
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Customization
    ax.set_title("Dataset Usage Distribution in Articles", fontsize=18, fontweight="bold")
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=9)

    # Improve layout
    plt.tight_layout()
    plt.show()

# Function to plot the ethnicity box plot with p-values
def plot_ethnicity_boxplot_with_pvalues():
    # Create DataFrame
    df = pd.DataFrame(DATA_ETHNICITY_BOXPLOT)

    # Calculate p-values
    p_values = calculate_p_values(df, "Ethnicity", "Proportion (%)")
    print("Calculated p-values:", p_values)

    # Plotting the box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Ethnicity", y="Proportion (%)", data=df, showmeans=True,
                meanline=True, meanprops={"color": "red", "ls": "--", "lw": 2},
                boxprops={"facecolor": "lightgray", "edgecolor": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                medianprops={"color": "black", "lw": 2})

    # # Adding p-values to the plot
    # y_max = df["Proportion (%)"].max() + 5  # Start slightly above the highest value
    # y_step = 10  # Distance between each annotation
    # for (group1, group2), p_value in p_values.items():
    #     x1 = list(df["Ethnicity"].unique()).index(group1)
    #     x2 = list(df["Ethnicity"].unique()).index(group2)
    #     y = y_max
    #     plt.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, c="black")
    #     plt.text((x1 + x2) / 2, y + 1.5, f"p = {p_value:.2e}", ha='center', fontsize=10, color="black")
    #     y_max += y_step

    # Customization
    plt.title("Ethnic Makeup Distribution in Databases", fontsize=16, weight="bold")
    plt.ylabel("Proportion in Databases (%)", fontsize=14)
    plt.xlabel("", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Function to calculate p-values between groups
def calculate_p_values(df, group_column, value_column):
    """
    Calculate pairwise p-values using Mann-Whitney U test for all unique combinations of groups.
    """
    unique_groups = df[group_column].unique()
    p_values = {}
    for i, group1 in enumerate(unique_groups):
        for j, group2 in enumerate(unique_groups):
            if i < j:  # Avoid duplicate comparisons
                group1_data = df[df[group_column] == group1][value_column]
                group2_data = df[df[group_column] == group2][value_column]
                # Ensure non-identical group data
                if not group1_data.equals(group2_data):
                    _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                    p_values[(group1, group2)] = p_value
                else:
                    p_values[(group1, group2)] = 1  # Assign high p-value for identical groups
    return p_values


####################################################################################################
# MAIN
####################################################################################################
if __name__ == "__main__":
    plot_ethnicity_distribution()
    plot_papers_per_dataset()
    plot_gender_distribution_horizontal()
    plot_dataset_usage_distribution()
    plot_dataset_usage_distribution_3d()
    plot_ethnicity_boxplot_with_pvalues()