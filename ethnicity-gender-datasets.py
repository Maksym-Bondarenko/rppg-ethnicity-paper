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

    plt.title("Ethnicity Distribution in Public Datasets", fontsize=14, weight="bold")
    plt.ylabel("Proportion in Datasets (%)", fontsize=12)
    plt.xlabel("Ethnicity", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_combined_tree_diagram():
    """
    Plots a 'tree-like' diagram showing:
      1) Dataset usage in articles (left side) subdivided by Fitzpatrick color slices.
      2) Gender distribution (right side) as a 100% stacked bar.
      3) Ensures no duplicate Fitzpatrick slices (max 6) per dataset.
    """
    datasets = list(DATASET_INFO.keys())

    # Extract arrays
    papers = np.array([DATASET_INFO[ds]["Papers"] for ds in datasets])
    female = np.array([0 if DATASET_INFO[ds]["Female"] == "N/A" else DATASET_INFO[ds]["Female"] for ds in datasets])
    male = np.array([0 if DATASET_INFO[ds]["Male"] == "N/A" else DATASET_INFO[ds]["Male"] for ds in datasets])
    total = np.array([DATASET_INFO[ds]["Total"] for ds in datasets])

    # Avoid division by zero
    total[total == 0] = 1

    female_pct = (female / total * 100).astype(float)
    male_pct = (male / total * 100).astype(float)
    papers_pct = (papers / papers.max() * 100).astype(int)

    fig, ax = plt.subplots(figsize=(20, 12))

    # ---------------- LEFT side: usage bars subdivided by *unique* Fitzpatrick colors ----------------
    for i, ds in enumerate(datasets):
        eth_string = DATA_ETHNICITY_MAPPING.get(ds, "N/A")
        sub_ethnicities = parse_ethnicity_string(eth_string)

        # Collect all Fitzpatrick indices in one set
        # so we don't duplicate any color slice
        fitz_index_set = set()

        for sub_eth in sub_ethnicities:
            if sub_eth == "n/a":
                continue  # ignore
            # map_to_fitzpatrick_colors returns a list of color codes,
            # but let's modify it to return the *indices* for even better dedup
            # Or: we can keep your existing function, but we need the index logic.

            # We'll do a simpler approach: create a small function inline:
            indices_for_subeth = get_fitz_indices(sub_eth)  # See helper below
            fitz_index_set |= indices_for_subeth  # union them

        # If no indices found => no ethnicity => single neutral color
        if not fitz_index_set:
            combined_colors = ["#D3D3D3"]  # “no ethnicity info”
        else:
            combined_colors = [FITZPATRICK_SCALE[idx] for idx in sorted(fitz_index_set)]

        # Subdivide the bar by however many distinct color codes we have
        n_subs = len(combined_colors)
        for j, color_code in enumerate(combined_colors):
            frac_start = j / n_subs
            frac_end = (j + 1) / n_subs
            left_edge = -papers_pct[i] * frac_start
            bar_width = -papers_pct[i] * (frac_end - frac_start)

            ax.barh(
                i,
                bar_width,
                left=left_edge,
                color=color_code,
                edgecolor="black"
            )

            # Label the slice with I, II, III, etc.
            label = FITZ_ROMAN.get(color_code, "")
            x_center = left_edge + bar_width / 2
            if abs(bar_width) > 5:  # skip labeling if the segment is too narrow
                ax.text(
                    x_center,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="black"
                )

        # Show the # of papers to the left
        ax.text(
            -papers_pct[i] - 3, i,
            str(papers[i]),
            va="center", ha="right",
            fontsize=20, color="black"
        )

    # ---------------- RIGHT side: 100% stacked bars for female / male (or “N/A”) ----------------
    for i, ds in enumerate(datasets):
        if female[i] == 0 and male[i] == 0:
            ax.barh(
                i, 100,
                color="white",
                edgecolor="black",
                label="No Gender Info" if i == 0 else None
            )
        else:
            # female
            ax.barh(
                i, female_pct[i],
                color="#E41A1C",
                label="Female Subjects" if i == 0 else None
            )
            # male
            ax.barh(
                i, male_pct[i],
                left=female_pct[i],
                color="#377EB8",
                label="Male Subjects" if i == 0 else None
            )

    # Percentage text + total subject count
    for i, val_total in enumerate(total):
        if female[i] + male[i] > 0:
            ax.text(
                female_pct[i] / 2, i,
                f"{int(female_pct[i])}%",
                va="center", ha="center",
                fontsize=20, color="white"
            )
            ax.text(
                female_pct[i] + male_pct[i] / 2, i,
                f"{int(male_pct[i])}%",
                va="center", ha="center",
                fontsize=20, color="white"
            )
        else:
            ax.text(50, i, "N/A",
                    va="center", ha="center",
                    fontsize=20, color="black")
        ax.text(103, i,
                str(int(val_total)),
                va="center", ha="left",
                fontsize=20, color="black")

    # Y-axis labels
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=20)

    # Title / labeling
    ax.set_title(
        "Article Usage, Ethnicity (Fitzpatrick Scale) and Gender Distribution",
        fontsize=28, weight="bold"
    )
    ax.set_xlabel(
        "Percentage Scale (left side: usage subdivided by Fitzpatrick colors; right side: gender distribution)",
        fontsize=22
    )
    ax.set_ylabel("Analyzed Public Datasets", fontsize=22)
    ax.axvline(0, color="black", linewidth=3)  # vertical divider

    # Mirrored x-ticks
    xticks = np.linspace(0, 100, 6, dtype=int)
    ax.set_xticks(np.concatenate((-xticks[::-1][1:], xticks)))
    ax.set_xticklabels(
        [f"{abs(t)}%" for t in np.concatenate((-xticks[::-1][1:], xticks))],
        fontsize=18
    )

    # Legend for Fitzpatrick + female/male/no‐gender + no ethnicity
    fitz_handles = []
    for idx in range(1, 7):
        c = FITZPATRICK_SCALE[idx]
        label_text = f"Fitzpatrick {FITZ_ROMAN[c]}"
        fitz_handles.append(mpatches.Patch(facecolor=c, edgecolor="black", label=label_text))

    gender_handles = [
        mpatches.Patch(color="#E41A1C", label="Female"),
        mpatches.Patch(color="#377EB8", label="Male"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="No Gender Info"),
        mpatches.Patch(facecolor="#D3D3D3", edgecolor="black", label="No Ethnicity Info"),
    ]

    ax.legend(
        handles=fitz_handles + gender_handles,
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