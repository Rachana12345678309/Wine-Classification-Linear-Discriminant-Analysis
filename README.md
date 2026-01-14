# Wine Classification using Linear Discriminant Analysis (LDA)

A machine learning project demonstrating the application of Linear Discriminant Analysis (LDA) for dimensionality reduction and classification. Unlike PCA, which focuses on maximum variance, this project uses LDA to maximize the distance between different wine classes, ensuring better diagnostic performance for classification models.

## Overview

This project utilizes the Wine dataset to classify different cultivars based on their chemical composition (e.g., Alcohol, Malic Acid, Ash). We implement LDA to reduce the feature space while preserving as much class-discriminatory information as possible. The project concludes by comparing the performance of Logistic Regression and Decision Tree models on the reduced feature set.

## Dataset

- **Source:** Scikit-learn built-in Wine dataset.
- **Classes:** 3 types of wine (Class 0, Class 1, Class 2).
- **Features:** 13 chemical attributes including:
  - Alcohol
  - Malic acid
  - Ash
  - Alkalinity of ash
  - Magnesium
  - Total phenols
  - Flavanoids, etc.

## Objectives

- Understand the difference between **PCA (Unsupervised)** and **LDA (Supervised)** dimensionality reduction.
- Project 13 chemical features into a 2D space that maximizes the separation between the three wine classes.
- Handle data preprocessing using **StandardScaler**.
- Evaluate the impact of dimensionality reduction on classification algorithms like **Logistic Regression** and **Decision Trees**.

## Methods and Analysis

The project follows a standard supervised learning pipeline:

- **Data Preparation**
  - Standardizing the 13 features to ensure the LDA algorithm is not biased by different units of measurement.
  - Splitting the dataset into Training (80%) and Testing (20%) sets.

- **Linear Discriminant Analysis (LDA)**
  - Implementing `LinearDiscriminantAnalysis` from Scikit-Learn.
  - Reducing the components to $n-1$ classes (in this case, 2 components).
  - Transforming the training and test data into the new "LDA space."



- **Model Training & Comparison**
  - **Logistic Regression:** Achieved an impressive **95% accuracy** on the reduced dataset.
  - **Decision Tree Classifier:** Evaluated to compare how non-linear models handle LDA-transformed features.

## Results

By reducing the dimensions from 13 to 2, the model remained highly efficient and accurate. The LDA transformation successfully separated the three wine cultivars into distinct clusters, making the classification task straightforward for the Logistic Regression model.



## Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `numpy` and `pandas`: Data handling.
  - `matplotlib` and `seaborn`: Visualizing the LDA components.
  - `scikit-learn`: LDA, Logistic Regression, Decision Trees, and data scaling.
- **Environment:** Jupyter / Google Colab

## How to Run

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/wine-lda-classification.git
   cd wine-lda-classification

2. *Create and activate a virtual environment (optional but recommended):*
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. *Install dependencies:*
   pip install pandas numpy seaborn matplotlib scikit-learn

4. *Open the notebook:*
   jupyter notebook 25_LDA_Handson.ipynb
