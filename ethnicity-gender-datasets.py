from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import matplotlib as mpl
import pandas as pd
import numpy as np

####################################################################################################
# DATA
####################################################################################################

# Set consistent LaTeX-style font and size
mpl.rcParams.update({
    "font.family": "serif",        # Use serif font like in LaTeX
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,               # Matches 10pt LaTeX document
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16
})

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

# MONK skin tone scale (10 shades from lightest to darkest)
MONK_SCALE = {
    1: "#f6ede4",
    2: "#f3e7db",
    3: "#f7ead0",
    4: "#eadaba",
    5: "#d7bd96",
    6: "#a07e56",
    7: "#825c43",
    8: "#604134",
    9: "#3a312a",
    10: "#292420",
}
MONK_ROMAN = {idx: str(idx) for idx in MONK_SCALE}

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
    Return a set of Fitzpatrick indices (1..6) based on common ethnicity keywords.
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
    if "black" in lower_str or "african" in lower_str:
        idxs.update([5, 6])
    if "indian" in lower_str:
        idxs.update([4, 5])
    if "varying skin tones" in lower_str or "fitzpatrick" in lower_str:
        idxs.update(range(1, 7))

    return idxs

def get_monk_indices(eth_string):
    """
    Return a set of Monk tone indices (1..10) based on ethnicity keywords.
    """
    if not eth_string or eth_string.strip().lower() == "n/a":
        return set()

    lower_str = eth_string.lower()
    idxs = set()
    if "white" in lower_str:
        idxs.update([1, 2, 3])
    if "caucasian" in lower_str:
        idxs.update([1, 2, 3, 4])
    if "asian" in lower_str:
        idxs.update([3, 4, 5, 6])
    if "latino" in lower_str or "native american" in lower_str:
        idxs.update([4, 5, 6, 7])
    if "black" in lower_str or "african" in lower_str:
        idxs.update([7, 8, 9, 10])
    if "indian" in lower_str:
        idxs.update([5, 6, 7])
    if "varying skin tones" in lower_str or "fitzpatrick" in lower_str:
        idxs.update(range(1, 11))

    return idxs

def parse_ethnicity_string(eth_string):
    if not eth_string or eth_string.strip().lower() == "n/a":
        return ["n/a"]
    return [e.strip().lower() for e in eth_string.split(",")]

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


def plot_ethnicity_boxplot_with_pvalues(
    save: bool = False,
    fname: str = "ethnicity_boxplot.png",
) -> None:
    """
    Box-plot of dataset‐proportion by coarse ethnicity groups,
    annotated with pair-wise Mann-Whitney-U p-values.

    Parameters
    ----------
    save   : if True write the figure to *fname* at 600 dpi instead of displaying it
    fname  : output path (extension decides the file type – png, tiff, pdf, …)
    """
    # -------- data & stats ----------------------------------------------------
    df_eth = prepare_ethnicity_dataframe()
    p_vals = calculate_p_values(df_eth, "Ethnicity", "Proportion (%)")

    monk_map = {"White": "1–3", "Asian": "3–6",
                "Black & Latino": "4–10", "Others": "1–10"}
    df_eth["EthnicityLabel"] = df_eth["Ethnicity"].apply(
        lambda e: f"{e} (Monk {monk_map.get(e,'N/A')})"
    )

    groups     = list(df_eth["Ethnicity"].unique())          # order is fixed here
    group_lbls = df_eth["EthnicityLabel"].unique()

    # -------- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x="EthnicityLabel", y="Proportion (%)", data=df_eth, ax=ax,
        showmeans=True, meanline=True,
        meanprops=dict(color="red", ls="--", lw=2),
        boxprops=dict(facecolor="lightgray", edgecolor="black"),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        medianprops=dict(color="black", lw=2),
    )

    # p-values
    y_max = df_eth["Proportion (%)"].max() + 5
    offset = 5
    for (g1, g2), pv in p_vals.items():
        x1, x2 = groups.index(g1), groups.index(g2)
        y = y_max
        ax.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, c="black")

        if pv < 0.01:
            exp  = int(np.floor(np.log10(pv)))
            base = pv / 10 ** exp
            ptxt = rf"$p = {base:.1f}\times10^{{{exp}}}$"
        else:
            ptxt = rf"$p = {pv:.2f}$"

        ax.text((x1 + x2) / 2, y + 1.2, ptxt, ha="center", va="bottom", fontsize=16)
        y_max += offset

    # axes
    ax.set_ylabel("Proportion in Datasets (%)", fontsize=16)
    ax.set_xlabel("Ethnicity (Monk Scale)",     fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", ls="--", alpha=0.7)
    fig.tight_layout()

    # -------- output ----------------------------------------------------------
    if save:
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_combined_tree_diagram():
    """
    Plots a 'tree-like' diagram showing:
      - Left side: usage in articles (scaled so max usage=100, subdivided by Fitzpatrick color slices)
      - Right side: 100% stacked bar for female/male or N/A
      - Right text columns for: total number of articles (`n`, italic, purple) and
        total number of subjects (`m`, italic, green)
      - **Bars for datasets with no ethnicity remain visible (gray-colored, no labels)**
    """

    SUBJECTS_COLOR = "#006D2C"  # Dark green for total subjects (m)
    PAPERS_COLOR   = "#984EA3"  # Purple for total articles (n)
    GRAY_COLOR     = "#D3D3D3"  # **Light gray for missing ethnicity info**

    datasets = list(DATASET_INFO.keys())

    # Extract numeric arrays
    papers = np.array([DATASET_INFO[ds]["Papers"] for ds in datasets])  # number of articles
    female = np.array([0 if DATASET_INFO[ds]["Female"] == "N/A" else DATASET_INFO[ds]["Female"] for ds in datasets])
    male   = np.array([0 if DATASET_INFO[ds]["Male"]   == "N/A" else DATASET_INFO[ds]["Male"]   for ds in datasets])
    total  = np.array([DATASET_INFO[ds]["Total"]       for ds in datasets])
    total[total == 0] = 1  # avoid division by zero

    female_pct = (female / total) * 100.0
    male_pct   = (male   / total) * 100.0
    max_usage  = papers.max()

    fig, ax = plt.subplots(figsize=(20, 12))

    # ---------- LEFT side: Usage scaled to max = 100 ----------
    for i, ds in enumerate(datasets):
        eth_string = DATA_ETHNICITY_MAPPING.get(ds, "N/A")
        sub_ethnicities = parse_ethnicity_string(eth_string)

        fitz_index_set = set()
        for sub_eth in sub_ethnicities:
            if sub_eth == "n/a":
                continue
            fitz_index_set |= get_fitz_indices(sub_eth)

        usage_abs = papers[i]
        usage_scaled = usage_abs * (100.0 / max_usage)  # scale usage to 0..100

        #  If no ethnicity => plain gray-colored bar with NO LABELS
        if not fitz_index_set:
            ax.barh(
                i,
                -usage_scaled,  # Negative for left-side bars
                left=0,
                color=GRAY_COLOR,
                edgecolor="black",
                label="No Ethnicity Info" if i == 0 else None  # Label only once
            )
        else:
            combined_colors = [FITZPATRICK_SCALE[idx] for idx in sorted(fitz_index_set)]
            ethnicity_labels = [FITZ_ROMAN[FITZPATRICK_SCALE[idx]] for idx in sorted(fitz_index_set)]

            n_subs = len(combined_colors)
            for j, (color_code, label) in enumerate(zip(combined_colors, ethnicity_labels)):
                frac_start = j / n_subs
                frac_end   = (j + 1) / n_subs

                seg_start  = usage_scaled * frac_start
                seg_end    = usage_scaled * frac_end
                bar_width  = -(seg_end - seg_start)  # Negative for left-side bars
                left_edge  = -seg_start

                ax.barh(
                    i,
                    bar_width,
                    left=left_edge,
                    color=color_code,
                    edgecolor="black"
                )

                # Only add labels if there is ethnicity info
                if abs(bar_width) > 4 and label:
                    ax.text(
                        left_edge + bar_width / 2, i, label,
                        ha="center", va="center",
                        fontsize=18, color="black"
                    )

    # ---------- RIGHT side: Gender Distribution ----------
    for i, ds in enumerate(datasets):
        if female[i] == 0 and male[i] == 0:
            ax.barh(i, 100, color="white", edgecolor="black")
        else:
            ax.barh(i, female_pct[i], color="#DF65B0", edgecolor="black")
            ax.barh(i, male_pct[i], left=female_pct[i], color="#2C7FB8", edgecolor="black")

    # ---------- Right-side text for percentages, total articles (n), and total subjects (m) ----------
    for i, val_total in enumerate(total):
        # Female and Male %
        if female[i] + male[i] > 0:
            ax.text(female_pct[i] / 2, i, f"{int(female_pct[i])}%", ha="center", va="center", fontsize=18, color="white")
            ax.text(female_pct[i] + male_pct[i] / 2, i, f"{int(male_pct[i])}%", ha="center", va="center", fontsize=18, color="white")
        else:
            ax.text(50, i, "N/A", ha="center", va="center", fontsize=18, color="black")

        # **Place column headers for `n` and `m` centered above**
        ax.set_xlim(-120, 120)
        ax.set_ylim(-0.5, len(datasets)-0.5)

        if i == 0:
            ax.text(105, len(datasets) + 0, r"$\it{m}$", ha="center", va="bottom", fontsize=18, color="black")
            ax.text(115, len(datasets) + 0, r"$\it{n}$", ha="center", va="bottom", fontsize=18, color="black")

        # Total Subjects (m) in green
        ax.text(105, i, str(int(val_total)), ha="center", va="center", fontsize=18, color=SUBJECTS_COLOR)

        # Total Articles (n) in purple
        ax.text(115, i, str(papers[i]), ha="center", va="center", fontsize=18, color=PAPERS_COLOR)

    # ---------- Y-axis ----------
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=22)

    # ---------- X-axis ----------
    ax.set_xlabel("")  # Remove default label
    ax.axvline(0, color="black", linewidth=3)  # Vertical divider

    ax.set_xticks([-50, 0, 50])  # Only three ticks
    ax.set_xticklabels(["Number of analyzed Articles ($\it{n}$)", "", "%"], fontsize=22, ha="center")

    # ---------- Legend ----------
    ethnicity_handles = [
        mpatches.Patch(facecolor=GRAY_COLOR, edgecolor="black", label="No Ethnicity Info"),
        mpatches.Patch(facecolor="#DF65B0", edgecolor="black", label="Female"),
        mpatches.Patch(facecolor="#2C7FB8", edgecolor="black", label="Male"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="No Gender Info"),
        mpatches.Patch(facecolor=SUBJECTS_COLOR, edgecolor="black", label=r"$\it{m}$ subjects in analyzed datasets"),
        mpatches.Patch(facecolor=PAPERS_COLOR, edgecolor="black", label=r"$\it{n}$ datasets usages in analyzed articles"),
    ]

    # Restore the missing **ethnicity legend**
    fitz_handles = []
    for idx in range(1, 7):
        c = FITZPATRICK_SCALE[idx]
        label_text = f"Fitzpatrick {FITZ_ROMAN[c]}"
        fitz_handles.append(mpatches.Patch(facecolor=c, edgecolor="black", label=label_text))

    ax.legend(handles=fitz_handles + ethnicity_handles, loc="upper left", fontsize=18)

    plt.tight_layout()
    plt.show()


def plot_tree_diagram_monk(
    save: bool = False,
    fname: str = "tree_diagram_monk.png",
) -> None:
    """
    “Tree” diagram of dataset usage coloured by Monk skin-tone indices
    (left) and gender distribution (right), with subject/article counts.

    Parameters
    ----------
    save   : if True write the figure to *fname* at 600 dpi instead of displaying it
    fname  : output path (extension decides the file type – png, tiff, pdf, …)
    """
    SUBJ_CLR = "#006D2C"   # green  – total subjects  (m)
    PAPR_CLR = "#984EA3"   # purple – total articles  (n)
    GRAY     = "#D3D3D3"

    ds_names = list(DATASET_INFO.keys())
    papers   = np.array([DATASET_INFO[d]["Papers"] for d in ds_names])
    female   = np.array([0 if DATASET_INFO[d]["Female"] == "N/A" else DATASET_INFO[d]["Female"]
                         for d in ds_names])
    male     = np.array([0 if DATASET_INFO[d]["Male"]   == "N/A" else DATASET_INFO[d]["Male"]
                         for d in ds_names])
    total    = np.array([DATASET_INFO[d]["Total"] for d in ds_names])
    total[total == 0] = 1
    f_pct, m_pct = (female / total) * 100, (male / total) * 100
    max_use = papers.max()

    fig, ax = plt.subplots(figsize=(20, 12))

    # ---- left: Monk-scale usage bars ----------------------------------------
    for i, ds in enumerate(ds_names):
        eth_str = DATA_ETHNICITY_MAPPING.get(ds, "N/A")
        monk_set = {idx for part in parse_ethnicity_string(eth_str)
                         for idx in get_monk_indices(part)}
        scaled = papers[i] * 100 / max_use

        if not monk_set:     # no info → gray bar
            ax.barh(i, -scaled, left=0, color=GRAY, edgecolor="black",
                    label="No Ethnicity Info" if i == 0 else None)
        else:
            cols = [MONK_SCALE[idx] for idx in sorted(monk_set)]
            lbls = [MONK_ROMAN[idx] for idx in sorted(monk_set)]
            n    = len(cols)
            for j, (c, l) in enumerate(zip(cols, lbls)):
                start, end = j / n, (j + 1) / n
                width = -(scaled * (end - start))
                left  = -scaled * start
                ax.barh(i, width, left=left, color=c, edgecolor="black")
                if abs(width) > 4:
                    ax.text(left + width / 2, i, l, ha="center", va="center",
                            fontsize=18, color="black")

    # ---- right: gender stacked bar ------------------------------------------
    for i in range(len(ds_names)):
        if female[i] == 0 and male[i] == 0:
            ax.barh(i, 100, color="white", edgecolor="black")
        else:
            ax.barh(i, f_pct[i],              color="#DF65B0", edgecolor="black")
            ax.barh(i, m_pct[i], left=f_pct[i], color="#2C7FB8", edgecolor="black")

    # ---- percentages and n/m columns ----------------------------------------
    ax.set_xlim(-120, 120)
    ax.set_ylim(-0.5, len(ds_names) - 0.5)

    for i, tot in enumerate(total):
        if female[i] + male[i] > 0:
            ax.text(f_pct[i] / 2,               i, f"{int(f_pct[i])}%",
                    ha="center", va="center", fontsize=18, color="white")
            ax.text(f_pct[i] + m_pct[i] / 2,    i, f"{int(m_pct[i])}%",
                    ha="center", va="center", fontsize=18, color="white")
        else:
            ax.text(50, i, "N/A", ha="center", va="center", fontsize=18)

        if i == 0:
            ax.text(105, len(ds_names), r"$\it{m}$", ha="center", va="bottom", fontsize=18)
            ax.text(115, len(ds_names), r"$\it{n}$", ha="center", va="bottom", fontsize=18)

        ax.text(105, i, str(int(tot)),   ha="center", va="center", fontsize=18, color=SUBJ_CLR)
        ax.text(115, i, str(papers[i]),  ha="center", va="center", fontsize=18, color=PAPR_CLR)

    # ---- axes & legend -------------------------------------------------------
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names, fontsize=22)
    ax.axvline(0, color="black", lw=3)
    ax.set_xticks([-50, 0, 50])
    ax.set_xticklabels(["Number of analyzed Articles ($\\it{n}$)", "", "%"],
                       fontsize=22, ha="center")
    monk_handles = [mpatches.Patch(facecolor=MONK_SCALE[idx], edgecolor="black",
                                   label=f"Monk {MONK_ROMAN[idx]}")
                    for idx in sorted(MONK_SCALE)]
    extra = [mpatches.Patch(facecolor="#DF65B0", edgecolor="black", label="Female"),
             mpatches.Patch(facecolor="#2C7FB8", edgecolor="black", label="Male"),
             mpatches.Patch(facecolor=GRAY,     edgecolor="black", label="No Ethnicity Info"),
             mpatches.Patch(facecolor="white",  edgecolor="black", label="No Gender Info"),
             mpatches.Patch(facecolor=SUBJ_CLR, edgecolor="black",
                            label=r"$\it{m}$ subjects in analyzed datasets"),
             mpatches.Patch(facecolor=PAPR_CLR, edgecolor="black",
                            label=r"$\it{n}$ dataset usages in analyzed articles")]
    ax.legend(handles=monk_handles + extra, loc="upper left", fontsize=18)
    fig.tight_layout()

    # ---- output --------------------------------------------------------------
    if save:
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

####################################################################################################
# Supplementary Information Plots
####################################################################################################

dataset_counts = {
        "UBFC-rPPG": 52, "PURE": 34, "VIPL-HR": 27, "COHFACE": 22,
        "MMSE-HR": 14, "MAHNOB-HCI": 10, "UBFC-Phys": 6, "MR-Nirp": 5,
        "OBF": 4, "BH-rPPG": 4, "BUAA": 3, "MPSC-rPPG": 3,
        "ECG Fitness": 2, "V4V": 2, "TokyoTech": 2, "MERL": 2,
        "PFF": 1, "LGI-PPGI": 1, "DDPM": 1, "UCLA-rPPG": 1,
        "DEAP": 1, "VicarPPG": 1, "BP4D+": 1, "CCUHR": 1
    }

def plot_dataset_distribution(dataset_counts: dict,
                              save: bool = False,
                              fname: str = "dataset_donut.png") -> None:
    """
    Donut-chart of dataset usage.

    • Shows *percentages* inside slices (only if ≥ 5 %), not raw counts.
    • Builds a long discrete colour list from *tab20 / tab20b / tab20c* to avoid repeats.
    • When *save=True* the figure is written to *fname* at 600 dpi instead of
      being displayed interactively.

    Parameters
    ----------
    dataset_counts : dict
        Mapping of dataset-name → paper-count.
    save           : bool  (default *False*)
        If *True* export the chart to *fname* (600 dpi) and close the figure.
    fname          : str
        Output path; the extension decides the file type (png, tiff, pdf, …).
    """
    # ---- 1. sort data --------------------------------------------------------
    datasets, counts = zip(*sorted(dataset_counts.items(),
                                   key=lambda kv: kv[1],
                                   reverse=True))
    total = sum(counts)

    # ---- 2. build colour list ------------------------------------------------
    def _build_colour_list(n: int):
        pal = list(mpl.cm.get_cmap("tab20").colors)
        if n > len(pal): pal += list(mpl.cm.get_cmap("tab20b").colors)
        if n > len(pal): pal += list(mpl.cm.get_cmap("tab20c").colors)
        if n > len(pal):                                      # unlikely
            extra = mpl.cm.hsv(np.linspace(0, 1, n - len(pal)))
            pal += list(extra)
        return pal[:n]

    colors = _build_colour_list(len(datasets))

    # ---- 3. helper for % labels ---------------------------------------------
    _autopct = lambda pct: f"{pct:.1f}%" if pct >= 5 else ""

    # ---- 4. plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _, autotexts = ax.pie(
        counts,
        startangle=90, counterclock=False,
        colors=colors,
        wedgeprops=dict(width=0.35),
        autopct=_autopct, pctdistance=0.78
    )

    for t in autotexts:           # format % inside slices
        t.set_fontsize(9)
        t.set_color("white")
        t.set_weight("bold")

    # donut hole
    ax.add_artist(plt.Circle((0, 0), 0.68, fc="white"))
    ax.axis("equal")

    # legend
    legend_labels = [f"{d} ({c})" for d, c in zip(datasets, counts)]
    ax.legend(
        wedges, legend_labels,
        title="Datasets",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.35, 1),
        fontsize="small"
    )

    fig.tight_layout()

    # ---- 5. output -----------------------------------------------------------
    if save:
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


####################################################################################################
# Raters Plot

def plot_raters_datasets():
    subjects       = np.arange(1, 10)
    rater_b_scores = np.full_like(subjects, 3)
    rater_a_scores = np.array([4, np.nan, 4, np.nan, 2,
                               np.nan, 4, np.nan, np.nan])

    mpl.rcParams.update({
        "font.family":   "serif",
        "font.serif":   ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":     14,
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    # vertical black line wherever both raters have a value
    for x, yA, yB in zip(subjects, rater_a_scores, rater_b_scores):
        if not np.isnan(yA):
            ax.plot([x, x], [yA, yB], color="black", lw=1)

    # scatter points (“x” markers)
    ax.scatter(subjects, rater_a_scores, marker="x", s=80,
               linewidths=2, color="#1f77b4", label="Rater A (M.B.)")
    ax.scatter(subjects, rater_b_scores, marker="x", s=80,
               linewidths=2, color="#ff7f0e", label="Rater B (M.E.)")

    # axes, grid, legend
    ax.set_xlabel("Subject")
    ax.set_ylabel("Monk Skin Tone Score")
    ax.set_xticks(subjects)
    ax.set_ylim(1, 10)
    ax.grid(axis="y", ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    fig.savefig("monk_rater_comparison.png", dpi=600, bbox_inches="tight")

####################################################################################################
# MAIN
####################################################################################################

if __name__ == "__main__":
    plot_ethnicity_boxplot_with_pvalues(save=True)
    #plot_combined_tree_diagram()
    plot_tree_diagram_monk(save=True)
    plot_dataset_distribution(dataset_counts, save=True)
    plot_raters_datasets()