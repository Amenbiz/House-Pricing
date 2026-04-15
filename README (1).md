# 🏠 California Housing Price Prediction
### COM747 — Data Science and Machine Learning
**Group Assignment | Due: 24 April 2026**

---

## 📌 Project Overview

This project applies a full data science lifecycle to the **California Housing Prices dataset** to predict median house values across California census districts. We follow the **CRISP-DM** (Cross Industry Standard Process for Data Mining) methodology, covering data cleaning, exploratory data analysis, feature engineering, and machine learning modelling.

The project is submitted as:
- **Component 1:** A 4-page IEEE research paper (group submission)
- **Component 2:** A 4-minute individual video presentation + code PDF

---

## 👥 Team Members & Roles

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Amen Ibizugbe** | Team Lead & Modelling Lead | Project coordination, CRISP-DM oversight, ML modelling (LR, DT, RF), model evaluation, final paper editing |
| **Ashmi** | Data Preprocessing Lead | Missing value handling, categorical encoding, feature scaling, data pipeline |
| **Kiani** | EDA & Visualisation Lead | Distributions, correlation heatmap, scatter plots, outlier identification |
| **Shakirat** | Feature Engineering & Ethics Lead | New feature creation, feature selection, feature importance, ethics section |

---

## 📂 Repository Structure

```
california-housing-project/
│
├── data/
│   ├── housing.csv               ← original raw dataset
│   └── housing_clean.csv         ← cleaned dataset (output of Stage 1)
│
├── notebooks/
│   ├── stage1_eda.ipynb          ← Data Cleaning & EDA (Ashmir + Kiani)
│   ├── stage2_features.ipynb     ← Feature Engineering (Shakirat + Ashmir)
│   ├── stage3_modelling.ipynb    ← ML Modelling & Evaluation (Team Lead)
│   └── stage4_final.ipynb        ← Final integrated notebook
│
├── figures/
│   └── *.png                     ← all charts and visualisations
│
├── paper/
│   └── COM747_group_paper.docx   ← IEEE research paper draft
│
├── requirements.txt              ← Python libraries needed
└── README.md                     ← this file
```

---

## 📊 Dataset

- **Source:** [Kaggle — California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Origin:** 1990 US Census data
- **Records:** 20,640 districts
- **Features:** 10 (9 input features + 1 target variable)
- **Target Variable:** `median_house_value`

| Feature | Description |
|---------|-------------|
| `longitude` | East-west position of the district |
| `latitude` | North-south position of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total rooms across all households |
| `total_bedrooms` | Total bedrooms (207 missing values — median imputed) |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median income (scaled units) |
| `ocean_proximity` | Categorical — distance from ocean |
| `median_house_value` | **Target** — median house value in USD |
