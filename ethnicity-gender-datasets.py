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

# Define hatches/patterns for each ethnicity
ETHNICITY_PATTERNS = {
    "white": "--",  # Horizontal lines
    "asian": "//",  # Forward diagonal stripes
    "black": "\\\\",  # Backward diagonal stripes
    "latino": "..",  # Dots
    "native american": "xx",  # Crossed lines
    "varying skin tones": "++",
    "indian": "oo",
    "others": "**",
    "n/a": ""  # No pattern
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

def parse_ethnicity_string(eth_string):
    """
    Safely splits the 'eth_string' on commas (and removes parentheses)
    then matches each piece to one of our known ETHNICITY_PATTERNS keys.
    Returns a list of lowercased ethnicities, e.g. ["white","black","asian"].
    """
    # If not provided or N/A, return ["n/a"] so that we get consistent handling.
    if not eth_string or eth_string.strip().lower() == "n/a":
        return ["n/a"]

    # Lower-case everything and strip parentheses
    clean_str = eth_string.lower().replace("(", "").replace(")", "")
    # Split by comma
    parts = [p.strip() for p in clean_str.split(",")]

    # Our recognized substrings -> pattern keys
    known_keywords = [
        "white",
        "black",
        "asian",
        "latino",
        "native american",
        "varying skin tones",
        "indian",
        "others",
        "skin colors in fitzpatrick scale",
        "fitzpatrick"
    ]

    found_ethnicities = []
    for p in parts:
        # Check if any known_keywords appear in this piece
        matched = False
        for kw in known_keywords:
            # e.g. "white" in "european white"
            # or "native american" in "black, white, asian, latino, native american"
            if kw in p:
                # For "skin colors in fitzpatrick scale" or "fitzpatrick" => treat as "varying skin tones"
                if "fitzpatrick" in kw:
                    found_ethnicities.append("varying skin tones")
                else:
                    found_ethnicities.append(kw)
                matched = True
                break
        if not matched:
            # If we didn't find a known keyword, treat it as "others" or "n/a"
            found_ethnicities.append("others")

    # Remove duplicates while preserving order
    final_list = list(dict.fromkeys(found_ethnicities))
    return final_list

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

    plt.title("Ethnicity Distribution in Databases", fontsize=14, weight="bold")
    plt.ylabel("Proportion in Databases (%)", fontsize=12)
    plt.xlabel("Ethnicity", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_combined_tree_diagram():
    """
    Plots a 'tree-like' diagram showing:
      1) Dataset usage in articles (left side) with possible subdivisions for each ethnicity (patterns).
      2) Gender distribution (right side) as a 100% stacked bar.
    """
    datasets = list(DATASET_INFO.keys())
    papers = np.array([DATASET_INFO[ds]["Papers"] for ds in datasets])
    female = np.array([0 if DATASET_INFO[ds]["Female"] == "N/A" else DATASET_INFO[ds]["Female"] for ds in datasets])
    male = np.array([0 if DATASET_INFO[ds]["Male"] == "N/A" else DATASET_INFO[ds]["Male"] for ds in datasets])
    total = np.array([DATASET_INFO[ds]["Total"] for ds in datasets])

    # Avoid division-by-zero
    total[total == 0] = 1

    female_pct = (female / total * 100).astype(float)
    male_pct = (male / total * 100).astype(float)
    papers_pct = (papers / papers.max() * 100).astype(int)

    # Colors for "single" vs "multi" vs "N/A" (based on DATASET_INFO or parsing)
    MONO_COLOR = "#44AA99"  # Green
    MULTI_COLOR = "#CC6677"  # Reddish
    NA_COLOR = "gray"

    fig, ax = plt.subplots(figsize=(20, 12))

    # -- LEFT side bars for "papers" (mirrored as negative x-values) --
    for i, ds in enumerate(datasets):
        # Parse the ethnicity string to a list so we can show multiple patterns
        eth_string = DATA_ETHNICITY_MAPPING.get(ds, "N/A")
        ethnicity_list = parse_ethnicity_string(eth_string)
        n_ethnicities = len(ethnicity_list)

        # Decide base color from the "MultiEthnicity" flag or number of ethnicities
        if "n/a" in ethnicity_list:
            base_color = NA_COLOR
        else:
            # If the dataset is known to have more than one distinct group
            if DATASET_INFO[ds]["MultiEthnicity"] or n_ethnicities > 1:
                base_color = MULTI_COLOR
            else:
                base_color = MONO_COLOR

        # Subdivide the left bar for each ethnicity in 'ethnicity_list'
        # so each portion has its own hatch pattern
        for j, ethnicity in enumerate(ethnicity_list):
            bar_frac_start = j / n_ethnicities
            bar_frac_end = (j + 1) / n_ethnicities
            # Convert fraction to negative width on the left side
            left_edge = -papers_pct[i] * bar_frac_start
            bar_width = -papers_pct[i] * (bar_frac_end - bar_frac_start)

            ax.barh(
                i,
                bar_width,
                left=left_edge,
                color=base_color,
                hatch=ETHNICITY_PATTERNS.get(ethnicity, ""),
                edgecolor="black"
            )

        # Add a text label with the actual #papers on the left side
        ax.text(-papers_pct[i] - 3, i, f"{papers[i]}", va="center", ha="right",
                fontsize=10, color="black")

    # -- RIGHT side 100% stacked bar for female+male or "N/A" --
    for i, ds in enumerate(datasets):
        if female[i] == 0 and male[i] == 0:
            # Gray bar for unknown
            ax.barh(i, 100, color="gray", label="No Gender Info" if i == 0 else None)
        else:
            # Pinkish bar for female
            ax.barh(i, female_pct[i], color="#CC6677", label="Female Subjects" if i == 0 else None)
            # Blueish bar for male
            ax.barh(i, male_pct[i], left=female_pct[i], color="#88CCEE",
                    label="Male Subjects" if i == 0 else None)

    # Add percentage labels for female/male or "N/A"
    for i, val_total in enumerate(total):
        if female[i] + male[i] > 0:
            ax.text(female_pct[i] / 2, i, f"{int(female_pct[i])}%", va="center", ha="center",
                    fontsize=9, color="white")
            ax.text(female_pct[i] + male_pct[i] / 2, i, f"{int(male_pct[i])}%", va="center",
                    ha="center", fontsize=9, color="white")
        else:
            ax.text(50, i, "N/A", va="center", ha="center", fontsize=9, color="white")
        # Also put total subject count out to the right
        ax.text(103, i, f"{int(val_total)}", va="center", ha="left", fontsize=10, color="black")

    # Y-axis labels = dataset names
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=12)

    # Title and x-axis label
    ax.set_title("Tree-like Diagram: Article Usage, Ethnicity and Gender Distribution",
                 fontsize=18, weight="bold")
    ax.set_xlabel(
        "Percentage Scale (left side: dataset usage in articles; right side: gender distribution in datasets)",
        fontsize=14)

    # Draw a thick vertical line at x=0 as a divider
    ax.axvline(0, color="black", linewidth=3)

    # Set mirrored x-ticks: from -100% up to +100%
    xticks = np.linspace(0, 100, 6, dtype=int)  # 0, 20, 40, 60, 80, 100
    ax.set_xticks(np.concatenate((-xticks[::-1][1:], xticks)))
    ax.set_xticklabels([f"{abs(t)}%" for t in np.concatenate((-xticks[::-1][1:], xticks))],
                       fontsize=10)

    # Build legends
    # 1) Single vs Multi vs No Ethnicity color
    ethnicity_handles = [
        mpatches.Patch(color=MONO_COLOR, label="Single Ethnicity Dataset"),
        mpatches.Patch(color=MULTI_COLOR, label="Multi-Ethnicity Dataset"),
        mpatches.Patch(color=NA_COLOR, label="No Ethnicity Info")
    ]

    # 2) Pattern legend (white, asian, black, etc.)
    pattern_handles = []
    for eth_key, hatch_style in ETHNICITY_PATTERNS.items():
        if eth_key != "n/a":  # skip the "n/a" empty hatch
            # Capitalize it in a friendly way for the label
            nice_label = eth_key.title().replace("American", "American")
            patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                   hatch=hatch_style, label=nice_label)
            pattern_handles.append(patch)

    # 3) Gender legend
    gender_handles = [
        plt.Line2D([0], [0], color="#CC6677", lw=8, label="Female Subjects"),
        plt.Line2D([0], [0], color="#88CCEE", lw=8, label="Male Subjects"),
        plt.Line2D([0], [0], color="gray", lw=8, label="No Gender Info")
    ]

    # Combine them
    ax.legend(handles=ethnicity_handles + pattern_handles + gender_handles,
              loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.show()

####################################################################################################
# MAIN
####################################################################################################

if __name__ == "__main__":
    plot_ethnicity_boxplot_with_pvalues()
    plot_combined_tree_diagram()