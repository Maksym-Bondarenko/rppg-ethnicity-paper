from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

####################################################################################################
# DATA
####################################################################################################

# Ethnicity of subjects in dataset
DATA_ETHNICITY_MAPPING = {
    "UBFC-rPPG": "White",
    "PURE": "White",
    "VIPL-HR": "Asian",
    "COHFACE": "White",
    "MMSE-HR": "Varying Skin Tones",
    "MAHNOB-HCI": "White, Asian",
    "LGI-PPGI": "Caucasian",
    "MR-Nirp": "Caucasian, Asian, Indian",
    "PFF": "Asian",
    "OBF": "Caucasian, Asian",
    "UBFC-Phys": "White",
    "BP4D+": "Black, White, Asian, Latino, Native American",
    "TokyoTech": "Asian",
    "BUAA-MIHR": "Asian",
    "CCUHR": "Asian",
    "MPSC-rPPG": "N/A",
    "BH-rPPG": "Asian",
    "V4V": "Black, White, Asian, Latino, Native American",
    "DDPM": "N/A",
    "UCLA-rPPG": "Fitzpatrick Scale 1 - 6",
    "DEAP": "N/A",
    "Vicar-PPG": "N/A",
    "MERL": "Varying Skin Tones",
    "ECG Fitness": "White"
}

# Update Proportions and Mapping for "Black & Latino" Category (used only for boxplots)
PROPORTIONS = {
    "White": [50, 55, 40, 60],
    "Asian": [30, 35, 25, 40],
    "Black & Latino": [20, 15, 10, 25],
    "Others": [10, 20, 15, 30],
}

# Fitzpatrick scale color codes (I through VI).
FITZPATRICK_SCALE = {
    1: "#FDECE0",  # Very fair
    2: "#F7D8BF",  # Fair / Light
    3: "#ECC3A0",  # Light / Medium
    4: "#CE9F7C",  # Moderately brown
    5: "#8E5B3F",  # Dark brown
    6: "#5B3F2A",  # Deeply pigmented
}

# Foe easier mapping and labeling on the plot
FITZ_ROMAN = {
    "#FDECE0": "I",
    "#F7D8BF": "II",
    "#ECC3A0": "III",
    "#CE9F7C": "IV",
    "#8E5B3F": "V",
    "#5B3F2A": "VI",
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
        "Female": 1,
        "Male": 6,
        "Total": 7
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

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################

def get_fitz_indices(eth_string):
    """
    Return a set of Fitzpatrick indices (1..6) for the given substring
    rather than a list of color codes, so we can unify them easily.
    """
    if not eth_string or eth_string.strip().lower() == "n/a":
        return set()

    lower_str = eth_string.lower()
    idxs = set()
    if "white" in lower_str:
        idxs.update([1, 2])
    if "caucasian" in lower_str:
        idxs.update([1, 2, 3])
    if "asian" in lower_str:
        idxs.update([2, 3, 4])
    if "latino" in lower_str or "native american" in lower_str:
        idxs.update([3, 4, 5])
    if "black" in lower_str:
        idxs.update([5, 6])
    if "indian" in lower_str:
        idxs.update([4, 5])
    if "varying skin tones" in lower_str:
        idxs.update(range(1, 7))
    if "fitzpatrick scale" in lower_str:
        idxs.update(range(1, 7))

    return idxs

def parse_ethnicity_string(eth_string):
    """
    Splits on commas and returns a list of lowercased partial strings,
    e.g. 'White, Asian' => ['white','asian'].
    """
    if not eth_string or eth_string.strip().lower() == "n/a":
        return ["n/a"]
    parts = [p.strip().lower() for p in eth_string.split(",")]
    return parts

def prepare_ethnicity_dataframe():
    """
    For the separate boxplot example, group all datasets into
    broad categories: White, Asian, Black & Latino, Others.
    Then generate a DataFrame to allow a boxplot of their 'proportions'.
    """
    ethnicity_data = group_ethnicities(DATA_ETHNICITY_MAPPING)
    df_ethnicity = pd.DataFrame(ethnicity_data)

    # Dynamically assign proportions to match the length of the DataFrame
    proportions = []
    for ethnicity in df_ethnicity["Ethnicity"].unique():
        n = len(df_ethnicity[df_ethnicity["Ethnicity"] == ethnicity])
        group_props = PROPORTIONS[ethnicity]  # e.g. [20,15,10,25] etc.

        # Replicate them to match how many rows we have
        # so that we can have a list of the same length as df
        repeated = []
        while len(repeated) < n:
            repeated.extend(group_props)
        # slice to exactly the needed length n
        repeated = repeated[:n]
        proportions.extend(repeated)

    df_ethnicity["Proportion (%)"] = proportions
    return df_ethnicity

def group_ethnicities(mapping):
    """
    Classify each dataset into one of four categories:
      1) Black & Latino (if text contains "black", "latino", or "native american")
      2) Asian         (if text contains "asian")
      3) White         (if text contains "white", "european", "caucasian")
      4) Others        (anything else, including "n/a", "fitzpatrick", etc.)

    IMPORTANT: The order of checks matters. If the string says "Black, White, Asian",
               it will be assigned "Black & Latino" first, because we prioritize that
               check below. Adjust the order to suit your exact grouping preference.
    """
    grouped = []
    for dataset, eth_str in mapping.items():
        if not eth_str:
            eth_str = "N/A"
        lower_str = eth_str.lower()

        # Priority #1: If it has 'black' or 'latino' or 'native american',
        #     put it in the Black & Latino group.
        if any(x in lower_str for x in ["black", "latino", "native american"]):
            grouped.append({"Dataset": dataset, "Ethnicity": "Black & Latino"})

        # Otherwise #2: If it has 'asian', label as 'Asian'
        elif "asian" in lower_str:
            grouped.append({"Dataset": dataset, "Ethnicity": "Asian"})

        # Otherwise #3: If it has 'white', 'european', or 'caucasian', label as 'White'
        elif any(x in lower_str for x in ["white", "european", "caucasian"]):
            grouped.append({"Dataset": dataset, "Ethnicity": "White"})

        # Otherwise #4: everything else -> "Others"
        else:
            grouped.append({"Dataset": dataset, "Ethnicity": "Others"})

    return grouped

def calculate_p_values(df, group_column, value_column):
    """
    Calculate pairwise p-values using Mann-Whitney U test for all unique combinations of groups.
    Logs detailed info for debugging.
    """
    unique_groups = df[group_column].unique()
    p_values = {}
    for i, group1 in enumerate(unique_groups):
        for j, group2 in enumerate(unique_groups):
            if i < j:  # avoid duplicates
                group1_data = df[df[group_column] == group1][value_column]
                group2_data = df[df[group_column] == group2][value_column]
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative="two-sided")

                print(f"Comparing {group1} vs {group2}: p-value={p_value:.4g}")
                p_values[(group1, group2)] = p_value

    return p_values

####################################################################################################
# PLOTTING FUNCTIONS
####################################################################################################


def plot_ethnicity_boxplot_with_pvalues():
    df_ethnicity = prepare_ethnicity_dataframe()
    p_vals = calculate_p_values(df_ethnicity, "Ethnicity", "Proportion (%)")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Ethnicity",
        y="Proportion (%)",
        data=df_ethnicity,
        showmeans=True,
        meanline=True,
        meanprops={"color": "red", "ls": "--", "lw": 2},
        boxprops={"facecolor": "lightgray", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black", "lw": 2},
    )

    # Add p-values
    y_max = df_ethnicity["Proportion (%)"].max() + 5
    step = 5
    unique_groups = list(df_ethnicity["Ethnicity"].unique())
    for (g1, g2), pv in p_vals.items():
        x1 = unique_groups.index(g1)
        x2 = unique_groups.index(g2)
        y = y_max
        plt.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, c="black")
        plt.text((x1 + x2) * 0.5, y + 1.2, f"p={pv:.2e}", ha="center", va="bottom", fontsize=9)
        y_max += step

    # plt.title("Ethnicity Distribution in Public Datasets", fontsize=14, weight="bold")
    plt.ylabel("Proportion in Datasets (%)", fontsize=12)
    plt.xlabel("Ethnicity", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_combined_tree_diagram():
    """
    Plots a 'tree-like' diagram showing:
      1) Left side: usage in articles (scaled so max usage=100,
         subdivided by Fitzpatrick color slices).
      2) Right side: 100% stacked bar for female / male or N/A.
      3) Two columns on the far right: total # subjects (in green text),
         # papers (in purple text).
      4) Those text colors also appear in the legend as color patches.
    """

    # -- color definitions for the numeric columns --
    SUBJECTS_COLOR = "#006D2C"  # a dark green, visible on white
    PAPERS_COLOR   = "#984EA3"  # a purple that stands out

    datasets = list(DATASET_INFO.keys())

    # 1) Extract numeric arrays
    papers = np.array([DATASET_INFO[ds]["Papers"] for ds in datasets])  # usage
    female = np.array([0 if DATASET_INFO[ds]["Female"] == "N/A" else DATASET_INFO[ds]["Female"] for ds in datasets])
    male   = np.array([0 if DATASET_INFO[ds]["Male"]   == "N/A" else DATASET_INFO[ds]["Male"]   for ds in datasets])
    total  = np.array([DATASET_INFO[ds]["Total"]       for ds in datasets])
    total[total == 0] = 1  # avoid div by zero

    female_pct = (female / total) * 100.0
    male_pct   = (male   / total) * 100.0
    max_usage  = papers.max()  # largest # of articles in any dataset

    fig, ax = plt.subplots(figsize=(20, 12))

    # ---------- LEFT side: scaled usage so max => 100, subdivided by Fitzpatrick colors ----------
    for i, ds in enumerate(datasets):
        eth_string = DATA_ETHNICITY_MAPPING.get(ds, "N/A")
        sub_ethnicities = parse_ethnicity_string(eth_string)

        fitz_index_set = set()
        for sub_eth in sub_ethnicities:
            if sub_eth == "n/a":
                continue
            fitz_index_set |= get_fitz_indices(sub_eth)

        # If no ethnicity => single neutral color
        if not fitz_index_set:
            combined_colors = ["#D3D3D3"]  # No Ethnicity
        else:
            combined_colors = [FITZPATRICK_SCALE[idx] for idx in sorted(fitz_index_set)]

        usage_abs = papers[i]
        usage_scaled = usage_abs * (100.0 / max_usage)  # scale usage to 0..100

        n_subs = len(combined_colors)
        for j, color_code in enumerate(combined_colors):
            frac_start = j / n_subs
            frac_end   = (j + 1) / n_subs

            seg_start  = usage_scaled * frac_start
            seg_end    = usage_scaled * frac_end
            bar_width  = -(seg_end - seg_start)   # negative for left side
            left_edge  = -seg_start

            ax.barh(
                i,
                bar_width,
                left=left_edge,
                color=color_code,
                edgecolor="black"
            )

            label = FITZ_ROMAN.get(color_code, "")
            x_center = left_edge + bar_width / 2
            if abs(bar_width) > 4:
                ax.text(
                    x_center, i, label,
                    ha="center", va="center",
                    fontsize=16, color="black"
                )

    # ---------- RIGHT side: 100% stacked bars for female/male (or N/A) ----------
    for i, ds in enumerate(datasets):
        if female[i] == 0 and male[i] == 0:
            # No Gender Info
            ax.barh(
                i, 100,
                color="white",
                edgecolor="black",
                label="No Gender Info" if i == 0 else None
            )
        else:
            # Female
            ax.barh(
                i, female_pct[i],
                color="#DF65B0",
                edgecolor="black",
                label="Female Subjects" if i == 0 else None
            )
            # Male
            ax.barh(
                i, male_pct[i],
                left=female_pct[i],
                color="#2C7FB8",
                edgecolor="black",
                label="Male Subjects" if i == 0 else None
            )

    # ---------- Right-side text for female% / male%, plus columns for #subjects & #papers ----------
    for i, val_total in enumerate(total):
        # female% & male% in the middle of each color wedge
        if female[i] + male[i] > 0:
            ax.text(
                female_pct[i] / 2, i, f"{int(female_pct[i])}%",
                ha="center", va="center", fontsize=20, color="white"
            )
            ax.text(
                female_pct[i] + male_pct[i] / 2, i, f"{int(male_pct[i])}%",
                ha="center", va="center", fontsize=20, color="white"
            )
        else:
            ax.text(
                50, i, "N/A",
                ha="center", va="center", fontsize=20, color="black"
            )

        # Now place total # subjects (in green)
        ax.text(
            102, i, str(int(val_total)),
            ha="left", va="center",
            fontsize=20, color=SUBJECTS_COLOR
        )
        # And # papers (in purple)
        ax.text(
            115, i, str(papers[i]),
            ha="left", va="center",
            fontsize=20, color=PAPERS_COLOR
        )

    # ---------- Y-axis ----------
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=20)

    # ---------- Title and axis labels ----------
    # ax.set_title(
    #     "Article Usage, Ethnicity (Fitzpatrick Scale) and Gender Distribution",
    #     fontsize=28, weight="bold"
    # )
    # ax.set_xlabel(
    #     "Scaled Articles (left) vs. Gender % (right) — Both sides = 100",
    #     fontsize=22
    # )
    ax.set_ylabel("Analyzed Public Datasets", fontsize=22)

    ax.axvline(0, color="black", linewidth=3)  # vertical divider

    # ---------- X-axis ticks (mirrored) ----------
    step_size = 10
    usage_vals = np.arange(0, max_usage + 1, step_size)
    usage_positions = -usage_vals * (100.0 / max_usage)  # negative side
    right_ticks = np.arange(0, 101, 20)                  # 0..100 for gender

    all_ticks = np.concatenate([usage_positions[::-1], right_ticks[1:]])
    # e.g. [-100, -80, ... 0, 20, 40, 60, 80, 100]
    ax.set_xticks(all_ticks)

    tick_labels = []
    for t in all_ticks:
        if t < 0:
            # invert scaling to show actual usage
            usage_val = -t * (max_usage / 100.0)
            tick_labels.append(str(int(usage_val)))
        else:
            tick_labels.append(f"{int(t)}%")
    ax.set_xticklabels(tick_labels, fontsize=18)

    # ---------- Legend ----------
    fitz_handles = []
    for idx in range(1, 7):
        c = FITZPATRICK_SCALE[idx]
        label_text = f"Fitzpatrick {FITZ_ROMAN[c]}"
        fitz_handles.append(mpatches.Patch(facecolor=c, edgecolor="black", label=label_text))

    gender_handles = [
        mpatches.Patch(facecolor="#DF65B0", edgecolor="black", label="Female"),
        mpatches.Patch(facecolor="#2C7FB8", edgecolor="black", label="Male"),
        mpatches.Patch(facecolor="white",   edgecolor="black", label="No Gender Info"),
        mpatches.Patch(facecolor="#D3D3D3", edgecolor="black", label="No Ethnicity Info"),
    ]

    # Add handles for the right-column text colors
    columns_handles = [
        mpatches.Patch(facecolor=SUBJECTS_COLOR, edgecolor="black", label="Total # Subjects (Text)"),
        mpatches.Patch(facecolor=PAPERS_COLOR,   edgecolor="black", label="# Papers (Text)")
    ]

    ax.legend(
        handles=fitz_handles + gender_handles + columns_handles,
        loc="upper left",
        fontsize=18
    )

    plt.tight_layout()
    plt.show()


####################################################################################################
# MAIN
####################################################################################################

if __name__ == "__main__":
    plot_ethnicity_boxplot_with_pvalues()
    plot_combined_tree_diagram()