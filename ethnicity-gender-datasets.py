from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

####################################################################################################
# DATA
####################################################################################################

# Fill missing values with "N/A" and map ethnicity correctly
DATA_ETHNICITY_MAPPING = {
    "UBFC-rPPG": ["White"],
    "PURE": ["White"],
    "VIPL-HR": ["Asian"],
    "COHFACE": ["White"],
    "MMSE-HR": ["Varying Skin Tones"],
    "MAHNOB-HCI": ["White", "Asian"],
    "LGI-PPGI": ["Caucasian"],
    "MR-Nirp": ["Indian", "Caucasian", "Asian"],
    "PFF": ["N/A"],
    "OBF": ["Caucasian", "Asian"],
    "UBFC-Phys": ["White"],
    "BP4D+": ["Black", "White", "Asian", "Latino", "Native American"],
    "TokyoTech": ["Asian"],
    "CCUHR": ["N/A"],
    "MPSC-rPPG": ["N/A"],
    "BH-rPPG": ["N/A"],
    "BUAA": ["N/A"],
    "V4V": ["Black", "White", "Asian", "Latino", "Native American"],
    "DDPM": ["N/A"],
    "UCLA-rPPG": ["Varying Skin Tones"],
    "VicarPPG": ["N/A"],
    "ECG Fitness": ["N/A"],
    "MERL": ["Varying Skin Tones"],
    "DEAP": ["N/A"]
}

# Define patterns for each ethnicity
ETHNICITY_PATTERNS = {
        "White": "--",  # Horizontal lines
        "Asian": "//",  # Diagonal stripes
        "Black": "\\\\",  # Backward diagonal stripes
        "Caucasian": "||",  # Vertical stripes
        "Latino": "..",  # Dots
        "Native American": "xx",  # Crossed lines
        "N/A": "",  # No pattern
        "Varying Skin Tones": "++", # also 1-6 (2-6) on Fitzpatrick scale
        "Indian": "oo",
    }

DATASET_INFO = {
    "UBFC-rPPG": {
        "Papers": 52,
        "MultiEthnicity": False,
        "Female": 11,
        "Male": 31,
        "Total": 42
    },
    "PURE": {
        "Papers": 34,
        "MultiEthnicity": False,
        "Female": 2,
        "Male": 8,
        "Total": 10
    },
    "VIPL-HR": {
        "Papers": 27,
        "MultiEthnicity": False,
        "Female": 28,
        "Male": 79,
        "Total": 107
    },
    "COHFACE": {
        "Papers": 22,
        "MultiEthnicity": False,
        "Female": 12,
        "Male": 28,
        "Total": 40
    },
    "MMSE-HR": {
        "Papers": 14,
        "MultiEthnicity": True,
        "Female": 23,
        "Male": 17,
        "Total": 40
    },
    "MAHNOB-HCI": {
        "Papers": 10,
        "MultiEthnicity": False,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 27
    },
    "UBFC-Phys": {
        "Papers": 6,
        "MultiEthnicity": False,
        "Female": 46,
        "Male": 10,
        "Total": 56
    },
    "MR-Nirp": {
        "Papers": 5,
        "MultiEthnicity": True,
        "Female": 2,
        "Male": 16,
        "Total": 18
    },
    "OBF": {
        "Papers": 4,
        "MultiEthnicity": False,
        "Female": 39,
        "Male": 61,
        "Total": 100
    },
    "BH-rPPG": {
        "Papers": 4,
        "MultiEthnicity": False,
        "Female": 1,
        "Male": 11,
        "Total": 12
    },
    "BUAA-MIHR": {
        "Papers": 3,
        "MultiEthnicity": False,
        "Female": 3,
        "Male": 12,
        "Total": 15
    },
    "MPSC-rPPG": {
        "Papers": 3,
        "MultiEthnicity": False,
        "Female": 2,
        "Male": 6,
        "Total": 8
    },
    "ECG-Fitness": {
        "Papers": 2,
        "MultiEthnicity": False,
        "Female": 3,
        "Male": 14,
        "Total": 17
    },
    "V4V": {
        "Papers": 2,
        "MultiEthnicity": True,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 179
    },
    "TokyoTech": {
        "Papers": 2,
        "MultiEthnicity": False,
        "Female": 1,
        "Male": 8,
        "Total": 9
    },
    "MERL": {
        "Papers": 2,
        "MultiEthnicity": True,
        "Female": 3,
        "Male": 9,
        "Total": 12
    },
    "PFF": {
        "Papers": 1,
        "MultiEthnicity": False,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 13
    },
    "LGI-PPGI": {
        "Papers": 1,
        "MultiEthnicity": False,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 25
    },
    "DDPM": {
        "Papers": 1,
        "MultiEthnicity": False,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 70
    },
    "UCLA-rPPG": {
        "Papers": 1,
        "MultiEthnicity": True,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 102
    },
    "DEAP": {
        "Papers": 1,
        "MultiEthnicity": False,
        "Female": 16,
        "Male": 16,
        "Total": 32
    },
    "VicarPPG": {
        "Papers": 1,
        "MultiEthnicity": False,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 10
    },
    "BP4D+": {
        "Papers": 1,
        "MultiEthnicity": True,
        "Female": 82,
        "Male": 58,
        "Total": 140
    },
    "CCUHR": {
        "Papers": 1,
        "MultiEthnicity": False,
        "Female": "N/A",
        "Male": "N/A",
        "Total": 22
    }
}

# Ethnicity Proportion Data
DATA_ETHNICITY_BOXPLOT = {
    "Ethnicity": ["White", "White", "Asian", "Asian", "Black", "Latino", "Others", "White", "Asian", "Others"],
    "Proportion (%)": [40, 45, 35, 30, 25, 20, 15, 50, 40, 10]
}

####################################################################################################
# PLOT-FUNCTIONS
####################################################################################################

# Combine datasets into broader categories with unique rows
def group_ethnicities_unique(mapping):
    grouped = []
    for dataset, ethnicities in mapping.items():
        # Map broader groups for simplicity
        if any(e in ["White", "Caucasian", "European"] for e in ethnicities):
            grouped.append({"Dataset": dataset, "Ethnicity": "White"})
        elif any(e == "Asian" for e in ethnicities):
            grouped.append({"Dataset": dataset, "Ethnicity": "Asian"})
        elif any(e == "Black" for e in ethnicities):
            grouped.append({"Dataset": dataset, "Ethnicity": "Black"})
        else:
            # Include "N/A" and all other unspecified ethnicities in "Others"
            grouped.append({"Dataset": dataset, "Ethnicity": "Others"})
    return grouped

# Create a DataFrame for box plot
ethnicity_data_unique = group_ethnicities_unique(DATA_ETHNICITY_MAPPING)
df_ethnicity = pd.DataFrame(ethnicity_data_unique)

# Ensure the proportions match the number of rows in the DataFrame
proportions = [40, 45, 35, 30, 25, 20, 15, 50, 40, 10, 15, 20, 35, 30, 10, 25, 35, 45, 50, 40, 30, 25, 35, 45, 20]
df_ethnicity["Proportion (%)"] = proportions[:len(df_ethnicity)]


# Function to calculate p-values between groups with higher precision
def calculate_p_values(df, group_column, value_column):
    """
    Calculate pairwise p-values using Mann-Whitney U test for all unique combinations of groups.
    Logs detailed information for debugging.
    """
    unique_groups = df[group_column].unique()
    p_values = {}
    for i, group1 in enumerate(unique_groups):
        for j, group2 in enumerate(unique_groups):
            if i < j:  # Avoid duplicate comparisons
                group1_data = df[df[group_column] == group1][value_column]
                group2_data = df[df[group_column] == group2][value_column]
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")

                # Log debugging information
                print(f"Comparing {group1} (n={len(group1_data)}) vs {group2} (n={len(group2_data)})")
                print(f"Group1 data: {list(group1_data)}")
                print(f"Group2 data: {list(group2_data)}")
                print(f"p-value: {p_value:.2e}\n")

                p_values[(group1, group2)] = p_value
    return p_values


# Function to plot the ethnicity box plot with p-values
def plot_ethnicity_boxplot_with_pvalues():
    p_values = calculate_p_values(df_ethnicity, "Ethnicity", "Proportion (%)")
    print("Calculated p-values:", p_values)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Ethnicity", y="Proportion (%)", data=df_ethnicity, showmeans=True,
                meanline=True, meanprops={"color": "red", "ls": "--", "lw": 2},
                boxprops={"facecolor": "lightgray", "edgecolor": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                medianprops={"color": "black", "lw": 2})

    y_max = df_ethnicity["Proportion (%)"].max() + 5
    y_step = 5
    for (group1, group2), p_value in p_values.items():
        x1 = list(df_ethnicity["Ethnicity"].unique()).index(group1)
        x2 = list(df_ethnicity["Ethnicity"].unique()).index(group2)
        y = y_max
        plt.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, c="black")
        plt.text((x1 + x2) / 2, y + 1.5, f"p = {p_value:.2e}", ha="center", fontsize=10, color="black")
        y_max += y_step

    plt.title("Ethnic Makeup Distribution in Databases", fontsize=16, weight="bold")
    plt.ylabel("Proportion in Databases (%)", fontsize=14)
    plt.xlabel("Ethnicity", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_combined_tree_diagram():
    # Extract data from the updated structure
    datasets = list(DATASET_INFO.keys())
    papers = np.array([DATASET_INFO[dataset]["Papers"] for dataset in datasets])
    female = np.array([0 if DATASET_INFO[dataset]["Female"] == "N/A" else DATASET_INFO[dataset]["Female"] for dataset in datasets])
    male = np.array([0 if DATASET_INFO[dataset]["Male"] == "N/A" else DATASET_INFO[dataset]["Male"] for dataset in datasets])
    total = np.array([DATASET_INFO[dataset]["Total"] for dataset in datasets])
    ethnicity_mapping = DATA_ETHNICITY_MAPPING

    # Avoid division by zero for missing gender data
    total[total == 0] = 1

    # Calculate percentages
    female_pct = (female / total * 100).astype(float)
    male_pct = (male / total * 100).astype(float)
    papers_pct = (papers / papers.max() * 100).astype(int)

    # Normalize percentages to ensure bars always sum to 100%
    for i in range(len(female_pct)):
        if female_pct[i] + male_pct[i] > 100:
            male_pct[i] = 100 - female_pct[i]
        elif female_pct[i] + male_pct[i] < 100:
            male_pct[i] = 100 - female_pct[i]

    # Colors for monotonicity and multi-ethnicity
    MONOTONIC_COLOR = "#44AA99"  # Green for single ethnicity
    MULTI_ETHNICITY_COLOR = "#CC6677"  # Red for multi-ethnicity
    NA_COLOR = "gray"  # Gray for N/A

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 12))

    # Left bar for papers (negative values for left side)
    for i, dataset in enumerate(datasets):
        ethnicity_list = ethnicity_mapping.get(dataset, ["N/A"])
        n_ethnicities = len(ethnicity_list)

        # Determine the base color based on monotonicity
        if "N/A" in ethnicity_list:
            base_color = NA_COLOR
        elif n_ethnicities == 1:
            base_color = MONOTONIC_COLOR
        else:
            base_color = MULTI_ETHNICITY_COLOR

        # Divide the bar into sections based on the number of ethnicities
        for j, ethnicity in enumerate(ethnicity_list):
            bar_start = -papers_pct[i] * (j / n_ethnicities)
            bar_end = -papers_pct[i] * ((j + 1) / n_ethnicities)
            ax.barh(
                i,
                bar_end - bar_start,
                left=bar_start,
                color=base_color,
                hatch=ETHNICITY_PATTERNS.get(ethnicity, ""),
                edgecolor="black",
                label=ethnicity if i == 0 or ethnicity not in ethnicity_list[:i] else None,
            )

        # Add the label for the number of articles
        ax.text(-papers_pct[i] - 5, i, f"{papers[i]}", va="center", ha="right", fontsize=10, color="black")

    # Right stacked bar for gender distribution (normalized to 100%)
    for i, dataset in enumerate(datasets):
        if female[i] == 0 and male[i] == 0:
            # No gender data, use one color for the entire bar
            ax.barh(i, 100, color="gray", label="No Gender Info" if i == 0 else None)
        else:
            ax.barh(i, female_pct[i], color="#CC6677", label="Female Subjects" if i == 0 else None)
            ax.barh(i, male_pct[i], left=female_pct[i], color="#88CCEE", label="Male Subjects" if i == 0 else None)

    # Adding percentage and total participants for each dataset on the right
    for i, total_val in enumerate(total):
        if female[i] + male[i] > 0:
            ax.text(female_pct[i] / 2, i, f"{int(female_pct[i])}%", va="center", ha="center", fontsize=9, color="white")
            ax.text(female_pct[i] + male_pct[i] / 2, i, f"{int(male_pct[i])}%", va="center", ha="center", fontsize=9, color="white")
        else:
            ax.text(50, i, "N/A", va="center", ha="center", fontsize=9, color="white")
        ax.text(105, i, f"{total_val}", va="center", ha="left", fontsize=10, color="black")

    # Customize
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=12)
    ax.set_title("Tree-like Diagram: Article Usage, Ethnicity and Gender Distribution", fontsize=18, weight="bold")
    ax.set_xlabel("Percentage Scale (left side: dataset usage in articles; right side: gender distribution in datasets)", fontsize=14)
    ax.axvline(0, color="black", linewidth=3)  # Central divider with larger width

    # X-axis percentages
    xticks = np.linspace(0, 100, 6, dtype=int)  # 0-100% scale for both sides
    ax.set_xticks(np.concatenate((-xticks[::-1][1:], xticks)))  # Mirror ticks for left and right
    ax.set_xticklabels([f"{abs(t)}%" for t in np.concatenate((-xticks[::-1][1:], xticks))], fontsize=10)

    # Add legends
    # Ethnicity color legend
    ethnicity_handles = [
        mpatches.Patch(color=MONOTONIC_COLOR, label="Single Ethnicity Dataset"),
        mpatches.Patch(color=MULTI_ETHNICITY_COLOR, label="Multi-Ethnicity Dataset"),
        mpatches.Patch(color=NA_COLOR, label="No Ethnicity Info"),
    ]

    # Ethnicity pattern legend
    pattern_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=ETHNICITY_PATTERNS[ethnicity], label=ethnicity)
        for ethnicity in ETHNICITY_PATTERNS.keys() if ethnicity != "N/A"
    ]

    # Gender legend
    gender_handles = [
        plt.Line2D([0], [0], color="#CC6677", lw=4, label="Female Subjects"),
        plt.Line2D([0], [0], color="#88CCEE", lw=4, label="Male Subjects"),
        plt.Line2D([0], [0], color="gray", lw=4, label="No Gender Info"),
    ]

    # Combine all legends
    ax.legend(handles=ethnicity_handles + pattern_handles + gender_handles, loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.show()


####################################################################################################
# MAIN
####################################################################################################
if __name__ == "__main__":
    plot_ethnicity_boxplot_with_pvalues()
    # plot_combined_tree_diagram()