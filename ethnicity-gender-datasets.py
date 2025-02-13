from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

####################################################################################################
# DATA
####################################################################################################

# Externalized data for easier editing
# DATA_ETHNICITY_DISTRIBUTION = {
#     "Ethnicity": ["European (white)", "Asian (white)", "Skin tones (Fitzpatrick scale)",
#                   "Caucasian, Asian", "Black, White, Asian, Latino, Native American", "N/A"],
#     "Values": [141, 144, 142, 118, 80, 166]
# }

# Fill missing values with "N/A" and map ethnicity correctly
DATA_ETHNICITY_MAPPING = {
    "UBFC-rPPG": ["White"],
    "PURE": ["White"],
    "VIPL-HR": ["Asian"],
    "COHFACE": ["White"],
    "MMSE-HR": ["Fitzpatrick II", "Fitzpatrick III", "Fitzpatrick IV", "Fitzpatrick V", "Fitzpatrick VI"],
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
    "UCLA-rPPG": ["Fitzpatrick I", "Fitzpatrick II", "Fitzpatrick III", "Fitzpatrick IV", "Fitzpatrick V", "Fitzpatrick VI"],
    "VicarPPG": ["N/A"],
    "ECG Fitness": ["N/A"],
    "MERL": ["Varying Skin Tones"],
    "DEAP": ["N/A"]
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

# DATA_ETHNICITY_BOXPLOT = {
#     "Ethnicity": ["Asian", "Asian", "Asian", "Black", "Black", "Black",
#                   "Other", "Other", "Other", "White", "White", "White"],
#     "Proportion (%)": [10, 15, 12, 5, 7, 6, 20, 18, 22, 80, 85, 78]
# }

####################################################################################################
# PLOT-FUNCTIONS
####################################################################################################

# Function: Ethnicity distribution (donut chart)
# def plot_ethnicity_distribution():
#     ethnicities = DATA_ETHNICITY_DISTRIBUTION["Ethnicity"]
#     values = DATA_ETHNICITY_DISTRIBUTION["Values"]
#     colors = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499"]
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#     wedges, texts, autotexts = ax.pie(
#         values,
#         labels=ethnicities,
#         autopct=lambda p: f"{p:.0f}%",
#         startangle=140,
#         colors=colors,
#         pctdistance=0.85
#     )
#
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     fig.gca().add_artist(centre_circle)
#
#     for autotext in autotexts:
#         autotext.set_color("white")
#         autotext.set_fontsize(12)
#         autotext.set_fontweight("bold")
#
#     ax.set_title("Ethnicity Distribution Across Datasets", fontsize=18, fontweight='bold')
#     plt.setp(texts, size=12)
#     plt.tight_layout()
#     plt.show()

# Function: Papers per dataset (bar chart)
# def plot_papers_per_dataset():
#     datasets = DATA_PAPERS_PER_DATASET["Datasets"]
#     papers = DATA_PAPERS_PER_DATASET["Papers"]
#     multi_ethnicity = DATA_PAPERS_PER_DATASET["MultiEthnicity"]
#
#     colors = ["#44AA99" if dataset in multi_ethnicity else "#CC6677" for dataset in datasets]
#
#     fig, ax = plt.subplots(figsize=(16, 8))  # Increased width
#     x = np.arange(len(datasets))
#     bars = ax.bar(x, papers, color=colors, width=0.8)
#
#     for bar, paper_count in zip(bars, papers):
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(paper_count),
#                 ha='center', va='bottom', fontsize=10)
#
#     ax.set_title("Number of Papers Using Each Dataset", fontsize=18, fontweight="bold")
#     ax.set_ylabel("Number of Papers", fontsize=14)
#     ax.set_xlabel("Dataset", fontsize=14)
#     ax.set_xticks(x)
#     ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=12)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#
#     ax.legend(
#         handles=[
#             plt.Line2D([0], [0], color="#44AA99", lw=4, label="Multi-Ethnicity"),
#             plt.Line2D([0], [0], color="#CC6677", lw=4, label="Predominantly Single Ethnicity")
#         ],
#         fontsize=12,
#         loc="upper right"
#     )
#
#     plt.tight_layout()
#     plt.show()

# Function: Gender distribution (horizontal bar chart)
# def plot_gender_distribution_horizontal():
#     datasets = DATA_GENDER_DISTRIBUTION["Datasets"]
#     female = np.array([0 if x == "N/A" else x for x in DATA_GENDER_DISTRIBUTION["Female"]], dtype=int)
#     male = np.array([0 if x == "N/A" else x for x in DATA_GENDER_DISTRIBUTION["Male"]], dtype=int)
#     total = np.array([0 if x == "N/A" else x for x in DATA_GENDER_DISTRIBUTION["Total"]], dtype=int)
#
#     # Ensure lengths of all arrays match
#     if not (len(datasets) == len(female) == len(male) == len(total)):
#         raise ValueError("Datasets, Female, Male, and Total arrays must have the same length!")
#
#     # Sort datasets by total subjects (descending order)
#     sorted_indices = np.argsort(-total)  # Sort by total in descending order
#     datasets = np.array(datasets)[sorted_indices]
#     female = female[sorted_indices]
#     male = male[sorted_indices]
#     total = total[sorted_indices]
#
#     # Calculate percentages, handle cases with total == 0
#     female_pct = np.zeros_like(total, dtype=int)
#     male_pct = np.zeros_like(total, dtype=int)
#     for i in range(len(total)):
#         if total[i] > 0:
#             female_pct[i] = int((female[i] / total[i]) * 100)
#             male_pct[i] = int((male[i] / total[i]) * 100)
#
#     # Define top-used datasets to bold
#     top_used_datasets = ["UBFC", "PURE", "VIPL-HR", "COHFACE", "MMSE-HR", "MAHNOB-HCI"]
#
#     # Bold text styling for top-used datasets
#     styled_datasets = [
#         f"$\\bf{{{dataset}}}$" if dataset in top_used_datasets else dataset for dataset in datasets
#     ]
#
#     # Colors
#     colors = {"Female": "#88CCEE", "Male": "#CC6677"}
#
#     # Plotting
#     fig, ax = plt.subplots(figsize=(16, 10))  # Increased width
#     y = np.arange(len(datasets))
#
#     ax.barh(y, female, color=colors["Female"], label='Female')
#     ax.barh(y, male, left=female, color=colors["Male"], label='Male')
#
#     # Adding percentages to bars
#     for i in range(len(datasets)):
#         if total[i] > 0:  # Avoid division by zero
#             ax.text(female[i] / 2, i, f"{female_pct[i]}%", va='center', ha='center', fontsize=9, color="white")
#             ax.text(female[i] + male[i] / 2, i, f"{male_pct[i]}%", va='center', ha='center', fontsize=9, color="white")
#
#     # Customization
#     ax.set_yticks(y)
#     ax.set_yticklabels(styled_datasets, fontsize=12)
#     ax.set_title("Gender Distribution Across Datasets (Sorted by Total Subjects)", fontsize=18, fontweight="bold")
#     ax.set_xlabel("Number of Subjects", fontsize=14)
#     ax.set_ylabel("Dataset", fontsize=14)
#     ax.legend(fontsize=12)
#     ax.grid(axis='x', linestyle='--', alpha=0.7)
#
#     plt.tight_layout()
#     plt.show()

# def plot_dataset_usage_distribution():
#     # Data for dataset usage
#     datasets = DATA_PAPERS_PER_DATASET["Datasets"]
#     papers = DATA_PAPERS_PER_DATASET["Papers"]
#
#     # Colors for the pie chart
#     colors = sns.color_palette("pastel", len(datasets))
#
#     # Plotting the donut chart
#     fig, ax = plt.subplots(figsize=(10, 6))
#     wedges, texts, autotexts = ax.pie(
#         papers,
#         labels=datasets,
#         autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
#         startangle=140,
#         colors=colors,
#         pctdistance=0.85
#     )
#
#     # Create a donut by adding a circle in the center
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     fig.gca().add_artist(centre_circle)
#
#     # Customization
#     ax.set_title("Dataset Usage Distribution in Articles", fontsize=18, fontweight="bold")
#     plt.setp(autotexts, size=10, weight="bold")
#     plt.setp(texts, size=9)
#
#     # Improve layout
#     plt.tight_layout()
#     plt.show()

# # Function to calculate p-values between groups with higher precision
# def calculate_p_values(df, group_column, value_column):
#     """
#     Calculate pairwise p-values using Mann-Whitney U test for all unique combinations of groups.
#     """
#     unique_groups = df[group_column].unique()
#     p_values = {}
#     for i, group1 in enumerate(unique_groups):
#         for j, group2 in enumerate(unique_groups):
#             if i < j:  # Avoid duplicate comparisons
#                 group1_data = df[df[group_column] == group1][value_column]
#                 group2_data = df[df[group_column] == group2][value_column]
#                 # Perform Mann-Whitney U test
#                 _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
#                 p_values[(group1, group2)] = p_value
#     return p_values
#
# # Function to plot the ethnicity box plot with precise p-values
# def plot_ethnicity_boxplot_with_pvalues():
#     # Create DataFrame
#     df = pd.DataFrame(DATA_ETHNICITY_BOXPLOT)
#
#     # Calculate p-values
#     p_values = calculate_p_values(df, "Ethnicity", "Proportion (%)")
#     print("Calculated p-values:", p_values)
#
#     # Plotting the box plot
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(x="Ethnicity", y="Proportion (%)", data=df, showmeans=True,
#                 meanline=True, meanprops={"color": "red", "ls": "--", "lw": 2},
#                 boxprops={"facecolor": "lightgray", "edgecolor": "black"},
#                 whiskerprops={"color": "black"},
#                 capprops={"color": "black"},
#                 medianprops={"color": "black", "lw": 2})
#
#     # Adding p-values to the plot
#     y_max = df["Proportion (%)"].max() + 5  # Start slightly above the highest value
#     y_step = 10  # Distance between each annotation
#     for (group1, group2), p_value in p_values.items():
#         x1 = list(df["Ethnicity"].unique()).index(group1)
#         x2 = list(df["Ethnicity"].unique()).index(group2)
#         y = y_max
#         plt.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, c="black")
#         plt.text((x1 + x2) / 2, y + 1.5, f"p = {p_value:.2e}", ha='center', fontsize=10, color="black")
#         y_max += y_step
#
#     # Customization
#     plt.title("Ethnic Makeup Distribution in Databases", fontsize=16, weight="bold")
#     plt.ylabel("Proportion in Databases (%)", fontsize=14)
#     plt.xlabel("", fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()


def plot_combined_tree_diagram():
    # Extract data from the updated structure
    datasets = list(DATASET_INFO.keys())
    papers = np.array([DATASET_INFO[dataset]["Papers"] for dataset in datasets])
    female = np.array([0 if DATASET_INFO[dataset]["Female"] == "N/A" else DATASET_INFO[dataset]["Female"] for dataset in datasets])
    male = np.array([0 if DATASET_INFO[dataset]["Male"] == "N/A" else DATASET_INFO[dataset]["Male"] for dataset in datasets])
    total = np.array([DATASET_INFO[dataset]["Total"] for dataset in datasets])
    multi_ethnicity = [DATASET_INFO[dataset]["MultiEthnicity"] for dataset in datasets]

    # Avoid division by zero for missing gender data
    total[total == 0] = 1

    # Calculate percentages
    female_pct = (female / total * 100).astype(int, copy=False)
    male_pct = (male / total * 100).astype(int, copy=False)
    papers_pct = (papers / papers.max() * 100).astype(int)

    # Colors for multi-ethnicity and non-multi-ethnicity datasets
    multi_ethnicity_colors = ["#44AA99" if is_multi else "#CC6677" for is_multi in multi_ethnicity]

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 12))

    # Left bar for papers (negative values for left side)
    ax.barh(datasets, -papers_pct, color=multi_ethnicity_colors, label="Dataset Usage (Papers)")

    # Right stacked bar for gender distribution (percentage scale)
    for i, dataset in enumerate(datasets):
        if female[i] == 0 and male[i] == 0:
            # No gender data, use one color for the entire bar
            ax.barh(i, 100, color="gray", label="No Gender Info" if i == 0 else None)
        else:
            # Ensure total width is fixed at 100%
            ax.barh(i, female_pct[i], color="#88CCEE", label="Female Subjects" if i == 0 else None)
            ax.barh(i, male_pct[i], left=female_pct[i], color="#CC6677", label="Male Subjects" if i == 0 else None)

    # Adding percentage and total participants for each dataset on the right
    for i, total_val in enumerate(total):
        if female[i] + male[i] > 0:
            # Percentages in the middle of each bar
            ax.text(female_pct[i] / 2, i, f"{female_pct[i]}%", va='center', ha='center', fontsize=9, color="white")
            ax.text(female_pct[i] + male_pct[i] / 2, i, f"{male_pct[i]}%", va='center', ha='center', fontsize=9, color="white")
            # Total participants at the end of the bar
            ax.text(105, i, f"{total_val}", va='center', ha='left', fontsize=10, color="black")
        else:
            ax.text(105, i, f"{total_val}", va='center', ha='left', fontsize=10, color="black")

    # Adding labels for number of papers (left bars)
    for i, val in enumerate(papers):
        ax.text(-papers_pct[i] - 5, i, f"{val} papers", va='center', ha='right', fontsize=10, color="black")

    # Customize
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=12)
    ax.set_title("Tree-like Diagram: Article Usage, Ethnicity and Gender Distribution", fontsize=18, weight="bold")
    ax.set_xlabel("Percentage Scale (left side: dataset usage in articles; right side: gender distribution in datasets)", fontsize=14)
    ax.axvline(0, color='black', linewidth=3)  # Central divider with larger width

    # X-axis percentages
    xticks = np.linspace(0, 100, 6, dtype=int)  # 0-100% scale for both sides
    ax.set_xticks(np.concatenate((-xticks[::-1][1:], xticks)))  # Mirror ticks for left and right
    ax.set_xticklabels([f"{abs(t)}%" for t in np.concatenate((-xticks[::-1][1:], xticks))], fontsize=10)

    # Add legends
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#44AA99", lw=4, label="Multi-Ethnicity Dataset"),
            plt.Line2D([0], [0], color="#CC6677", lw=4, label="Single-Ethnicity Dataset"),
            plt.Line2D([0], [0], color="#88CCEE", lw=4, label="Female Subjects"),
            plt.Line2D([0], [0], color="#CC6677", lw=4, label="Male Subjects"),
            plt.Line2D([0], [0], color="gray", lw=4, label="No Gender Info"),
        ],
        loc="lower right",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


####################################################################################################
# MAIN
####################################################################################################
if __name__ == "__main__":
    # plot_ethnicity_distribution()
    # plot_papers_per_dataset()
    # plot_gender_distribution_horizontal()
    # plot_dataset_usage_distribution()
    # plot_ethnicity_boxplot_with_pvalues()
    plot_combined_tree_diagram()