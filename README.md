# Ethnicity Representation in rPPG Datasets


This repository contains Python scripts for analyzing and visualizing the **ethnicity representation** in publicly available **remote photoplethysmography (rPPG) datasets**. The visualizations help assess dataset diversity and potential biases in heart rate estimation research.

## 📌 Overview
- The code **processes metadata** from multiple rPPG datasets.
- It **categorizes dataset subjects** based on ethnicity, gender, and Fitzpatrick skin tones.
- It **generates statistical visualizations**, including:
  - **Ethnicity Distribution Boxplots**
  - **Tree-Like Gender & Ethnicity Composition Diagrams**
- The results are **used exclusively in a scientific paper** authored by **Maksym Bondarenko, Carlo Menon, and Mohamed Elgendi**.

## 📊 Data Sources
The dataset metadata analyzed in this study includes:
- **Ethnicity mappings** of subjects in rPPG datasets.
- **Fitzpatrick skin tone classification** for ethnicity groups.
- **Gender distributions** (where available).
- **Dataset usage in scientific literature**.

## 🛠 Dependencies
To run the visualizations, install the following Python packages:
```bash
pip install numpy pandas seaborn matplotlib scipy
```

## 🚀 Usage
Run the main script to generate the visualizations:

```bash
python main.py
```

Example Outputs:
- Ethnicity Distribution Boxplot with statistical significance markers.
- Combined Gender & Ethnicity Tree Diagram summarizing dataset diversity.
  
## 📄 Citation
If you use this repository or its results, please cite the following paper:


## 📧 Contact
For inquiries, contact the authors:

* Maksym Bondarenko – TUM, Munich, Germany (maksym.bondarenko@tum.de)
* Carlo Menon – ETH Zürich, Switzerland (carlo.menon@hest.ethz.ch)
* Mohamed Elgendi – Khalifa University, Abu Dhabi, UAE (mohamed.elgendi@ku.ac.ae)

## ⚠️ Disclaimer
This repository is intended solely for generating plots used in the above-mentioned publication. It does not provide raw dataset files or sensitive personal data.
