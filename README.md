# Diabetes Classifier
### Lab Overview

This lab evaluates 5 machine learning models (**KNN, Decision Tree, SVM, Gaussian NB, and MLP**) for binary **Diabetes Risk Prediction** using the Pima Indians dataset. After median imputation and standard scaling on a 70:30 split, performance was comparatively analyzed across key metrics including Accuracy, Recall, Specificity, F1-Score, ROC-AUC, and MCC.

The overall steps are detailed below:

# Step 1: Dataset Selection

The selected dataset is **Pima Indians Diabetes Dataset** which contains medical diagnostic measures for **768 female patients** of Pima Indian heritage. It includes **8 clinical features** (e.g., Glucose, BMI, Age) used to predict binary diabetes risk (`0` or `1`).

* **Source Link:** [Pima Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
---

# Step 2: Exploratory Data Analysis (EDA) & Preprocessing

### Step 2.1: Data Loading & Initial Exploration

1. **Dataset Overview & Structure:**
   - **Total Records:** 768 rows.
   - **Total Attributes:** 9 columns (8 continuous predictor features + 1 binary target variable `Outcome`).
   - **Data Types:** 7 integer columns (`int64`) and 2 floating-point columns (`float64`). There are no non-numeric categorical attributes, meaning categorical encoding (e.g., One-Hot Encoding) is not required.

2. **Duplicate Records Check:**
   - **Result:** `0` duplicate rows detected.
   - **Action:** No record removal is required for duplicates.

3. **Detection of Missing / Invalid Values Disguised as Zeros:**
   - **Observation:** Statistical summary (`describe()`) reveals minimum values of `0.0` in physiological parameters where zero is biologically impossible:
     - `Glucose` (min: 0)
     - `BloodPressure` (min: 0)
     - `SkinThickness` (min: 0)
     - `Insulin` (min: 0)
     - `BMI` (min: 0.0)
    - **Action Required:** These `0` values represent missing or unrecorded entries. In Step 2.2, they must be converted to `NaN` and imputed appropriately rather than removing the rows.
   - *Note:* `Pregnancies` can validly be `0` (indicating zero past pregnancies), so `0` values in this column are preserved.

4. **Target Class Breakdown (`Outcome`):**
   - **Class Distribution:** ~65.1% Non-Diabetic (`0`) vs. ~34.9% Diabetic (`1`).
   - **Insight:** Mild class imbalance exists in the target variable. Stratified train-test splitting will be applied in Step 2.3 to preserve this class proportion across both training and testing datasets.

### Step 2.2: Data Cleaning & Missing Value Imputation

1. **Identification of Invalid Zero Values:**
   - Analysis revealed missing data disguised as zeros across 5 physiological parameters:
     - `Insulin`: 374 missing values (~48.7% of the dataset)
     - `SkinThickness`: 227 missing values (~29.6% of the dataset)
     - `BloodPressure`: 35 missing values (~4.6%)
     - `BMI`: 11 missing values (~1.4%)
     - `Glucose`: 5 missing values (~0.7%)

2. **Justification for Imputation over Deletion:**
   - **Why not listwise deletion?** Deleting all rows containing zero values would discard almost half the dataset (over 370 instances). This would cause severe loss of statistical power and introduce selection bias.
   - **Why Grouped Median Imputation?** - Median is robust against extreme outliers (unlike the mean).
     - Grouping by target class (`Outcome`) ensures class-specific distributions are preserved. For instance, diabetic patients (`Outcome=1`) typically exhibit higher median glucose and insulin levels than non-diabetic patients (`Outcome=0`).

3. **Imputation Outcome:**
   - All `0` values in physiological features were successfully converted to `NaN` and imputed.
   - Post-imputation check verifies **0 missing values (`0 NaN`)** across all columns in `df_clean`.


### Step 2.3: Target Class Breakdown & Stratified Train-Test Split

1. **Target Class Imbalance Analysis:**
   - **Non-Diabetic (`Outcome = 0`):** 500 instances (65.1%)
   - **Diabetic (`Outcome = 1`):** 268 instances (34.9%)
   - **Observation:** The target variable shows a mild class imbalance (roughly 1.86:1 ratio of negative to positive cases). 

2. **Train-Test Splitting Strategy:**
   - **Split Ratio:** 80% Training (614 instances) and 20% Testing (154 instances).
   - **Justification for Stratified Sampling (`stratify=y`):**
     - Simple random sampling could inadvertently lead to an unequal distribution of diabetic cases between the train and test sets (e.g., test set receiving too few positive cases).
     - Applying `stratify=y` guarantees that both the 614 training samples and 154 testing samples preserve the exact **65.1% to 34.9%** class distribution ratio, preventing split-induced evaluation bias.


### Step 2.4: Feature Scaling Implementation

1. **Need for Feature Scaling:**
   - Features in the dataset have vastly different numeric ranges (e.g., `Insulin` values range into hundreds, while `DiabetesPedigreeFunction` remains < 2.5).
   - Distance-based models (such as **K-Nearest Neighbors**) and gradient-based models (**Multi-Layer Perceptron**, **SVM**) are highly sensitive to feature magnitudes. Unscaled features would cause columns with larger values to dominate distance calculations unfairly.

2. **Standardization Strategy (`StandardScaler`):**
   - We applied Z-score standardization ($\mu = 0, \sigma = 1$), centering all features around $0$ with unit variance.
   - **Data Leakage Prevention:** The scaler was **fitted strictly on `X_train`** and then used to transform both `X_train` and `X_test`. This ensures no information from the test dataset leaks into model training.

3. **Output Verification:**
   - The sample scaled values show standard normal distribution properties (values centered near $0$, mostly falling between $-3$ and $+3$), confirming successful transformation.


# Step 3: Model Development & Hyperparameter Tuning

### Step 3.1: KNN From Scratch

1. **Algorithm Implementation:**
   - Implemented a pure **K-Nearest Neighbors (KNN)** classifier from scratch using `NumPy` without relying on `scikit-learn`'s model classes.
   - **Distance Metric:** Utilized Euclidean Distance: 
     $$d(x_1, x_2) = \sqrt{\sum_{i=1}^{n} (x_{1,i} - x_{2,i})^2}$$
   - **Decision Rule:** For each test sample, the Euclidean distance to all 614 training instances was computed. The top $K=7$ nearest instances were selected, and majority voting (`bincount().argmax()`) determined the predicted class (`0` for non-diabetic, `1` for diabetic).

2. **Hyperparameter Selection:**
   - Selected $K = 7$ as an odd number to prevent tie votes in binary classification.

3. **Execution Verification:**
   - The algorithm successfully output class predictions and probability estimates (`predict_proba`) for all 154 test samples.


###  Step 3.2: Decision Tree Classifier

1. **Model Implementation:**
   - Implemented a **Decision Tree Classifier** using `scikit-learn` with the Gini Impurity criterion to measure node split quality.

2. **Overfitting Prevention & Pruning Strategy:**
   - Fully expanded decision trees tend to overfit by memorizing noise in the training set.
   - To control tree complexity, pre-pruning hyperparameter constraints were applied:
     - `max_depth=4`: Restricts the tree depth to prevent deep, over-specialized branches.
     - `min_samples_split=10`: Requires at least 10 samples at a node before a split can occur.

3. **Execution Verification:**
   - The model successfully evaluated the scaled test dataset and generated both discrete class predictions and predicted probabilities (`predict_proba`).

### Step 3.3: Support Vector Machine - SVM

1. **Model Implementation:**
   - Implemented a **Support Vector Machine (SVM)** classifier using `scikit-learn`.
   - **Kernel Selection:** Radial Basis Function (`rbf`) kernel was chosen to map non-linearly separable physiological features into a higher-dimensional space where an optimal linear separating hyperplane can be identified.
   - **Regularization ($C=1.0$):** Standard regularization strength to balance margin width and training misclassification errors.

2. **Feature Dependency:**
   - SVM relies heavily on distance computations between support vectors and decision boundaries; hence, the standardized features from Step 2.4 were essential for optimal hyperplane convergence.

3. **Deprecation Warning Note:**
   - Note on Scikit-Learn warning: The `probability=True` flag in `SVC` (which uses Platt scaling under the hood) was flagged for future deprecation in favor of `CalibratedClassifierCV`. However, probability output was enabled to facilitate ROC-AUC evaluation in Step 4.


### Step 3.4: Gaussian Naïve Bayes

1. **Model Implementation:**
   - Implemented a **Gaussian Naïve Bayes (GNB)** classifier using `scikit-learn`.
   - **Theoretical Basis:** GNB relies on Bayes' Theorem:
     $$P(Y|X) = \frac{P(X|Y) P(Y)}{P(X)}$$
     It assumes continuous features follow a normal (Gaussian) distribution and are conditionally independent given the class label.

2. **Suitability & Characteristics:**
   - **Baseline Efficiency:** GNB is computationally lightweight and highly efficient. 
   - **Robustness:** Despite the independence assumption being rarely strictly true in medical data (e.g., Glucose and Insulin correlate), GNB performs surprisingly well in binary clinical risk predictions.

### Step 3.5: Multi-Layer Perceptron - MLP

1. **Model Architecture & Configuration:**
   - Implemented a **Multi-Layer Perceptron (MLP)** feedforward neural network using `scikit-learn`.
   - **Architecture:** 2 hidden layers with 16 and 8 neurons respectively `(16, 8)`.
   - **Activation Function:** Rectified Linear Unit (`ReLU`) for non-linear feature transformation.
   - **Optimization:** `adam` stochastic gradient-based optimizer.

2. **Overfitting Prevention & Convergence Note:**
   - **Regularization ($L_2$ penalty):** Set `alpha=0.01` to penalize large weight coefficients and reduce potential overfitting.
   - **Convergence Behavior:** The optimization reached the maximum iteration limit (`max_iter=500`). This is typical for neural network training on non-linearly complex datasets without early stopping callbacks, but the loss curve flattened sufficiently to produce stable test predictions and class probabilities.


# Step 4: Model Evaluation and Performance Analysis

1. **Selection & Justification of Metrics:**
   - **Accuracy:** General metric for overall correctness across all predictions ($N=154$).
   - **Precision & Recall (Sensitivity):** Crucial in clinical diabetes diagnosis. High **Recall** ensures potential diabetic patients are not overlooked (minimizing False Negatives), while high **Precision** avoids unnecessary treatment anxiety (minimizing False Positives).
   - **Specificity (True Negative Rate):** Measures the proportion of actual non-diabetic individuals correctly identified.
   - **ROC-AUC (Area Under ROC Curve):** Evaluates overall class separability across various probability thresholds without being biased by threshold selection.
   - **Matthews Correlation Coefficient (MCC):** Evaluates binary prediction quality robustly even when accounting for proportional differences between classes ($[-1, +1]$ scale).

2. **Confusion Matrix Key Findings:**
   - **MLP Classifier** achieved the highest overall balance, correctly identifying $43/54$ true positive diabetic cases (Recall = $79.63\%$) and $91/100$ true negative cases (Specificity = $91.00\%$).
   - **Gaussian Naïve Bayes** exhibited lower overall precision and specificity due to strong conditional independence assumptions among physiological features (e.g., Glucose & Insulin correlation).


# Step 5: Comparative Analysis and Interpretation of Results

## 1. Summary Performance Metric Comparison Table

| Classifier | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom KNN** | 0.8052 | 0.7222 | 0.7222 | 0.8500 | 0.7222 | 0.8677 | 0.5722 |
| **Decision Tree** | 0.8377 | 0.7843 | 0.7407 | 0.8900 | 0.7619 | 0.8980 | 0.6395 |
| **Support Vector Machine (SVM)** | 0.8377 | 0.7636 | 0.7778 | 0.8700 | 0.7706 | 0.8974 | 0.6451 |
| **Gaussian Naïve Bayes** | 0.7273 | 0.6034 | 0.6481 | 0.7700 | 0.6250 | 0.8017 | 0.4118 |
| **Multi-Layer Perceptron (MLP)** | **0.8701** | **0.8269** | **0.7963** | **0.9100** | **0.8113** | **0.9144** | **0.7127** |

---

## 2. Best-Performing Classifier Identification

The **Multi-Layer Perceptron (MLP)** is unequivocally the **best-performing classifier** across all evaluated metrics:
* **Highest Accuracy ($87.01\%$):** Correctly classified 134 out of 154 test samples.
* **Highest F1-Score ($0.8113$) & MCC ($0.7127$):** Indicates exceptional balance between Precision and Recall on binary outcomes.
* **Highest Recall ($79.63\%$):** Detected the largest proportion of true positive diabetic cases (43 out of 54).
* **Highest ROC-AUC ($0.9144$):** Demonstrates superior discriminative capacity across all confidence thresholds.

---

## 3. Analysis and Interpretation of Results

### A. Why Certain Classifiers Performed Better or Worse
1. **Multi-Layer Perceptron (MLP) & Support Vector Machine (SVM):**
   - **Why they succeeded:** Both models excel at modeling complex, non-linear relationships and high-dimensional interactions between biological markers (e.g., non-linear interactions between BMI, Glucose, and Age). $L_2$ regularization (`alpha=0.01` in MLP) and the RBF kernel in SVM prevented overfitting.
2. **Decision Tree:**
   - **Why it performed well:** Pre-pruning (`max_depth=4`, `min_samples_leaf=5`) constrained tree depth, preserving generalization while capturing major hierarchical threshold splits in glucose levels.
3. **Custom KNN:**
   - **Why it was competitive ($80.52\%$):** Distance calculation on standard-scaled features ensured fair metric weighting across features, though local neighbor voting remains sensitive to density boundaries.
4. **Gaussian Naïve Bayes:**
   - **Why it performed worst ($72.73\%$):** GNB relies on the strong assumption that all input features are conditionally independent given the target class. In medical diagnostic data, features like `Glucose`, `Insulin`, `BMI`, and `Age` exhibit natural correlations, directly violating this independence assumption.

---

### B. Effect of Preprocessing Techniques
* **Missing Value Imputation:** Replacing zero-entry anomalies in biological features (e.g., Blood Pressure, Glucose) with median values prevented extreme skewness and erroneous distance metrics.
* **Standard Scaling:** Features like `Insulin` (range ~0–800) and `Pedigree Function` (range ~0.08–2.4) exist on vastly different scales. Standardizing features to mean = 0 and variance = 1 was essential for distance-based models (KNN), gradient-based models (MLP), and margin-based models (SVM).

---

### C. Algorithm Advantages and Limitations

| Algorithm | Key Advantages | Key Limitations |
| :--- | :--- | :--- |
| **Custom KNN** | Simple, interpretable non-parametric learning. | Computationally expensive during prediction; sensitive to noise. |
| **Decision Tree** | Highly interpretable decision rules. | Prone to high variance and step-wise decision boundaries. |
| **SVM (RBF)** | Effective in high-dimensional continuous spaces. | Memory-intensive; hyperparameter sensitive ($\mathcal{C}$, $\gamma$). |
| **Gaussian NB** | Fast training, robust to minor missing values. | Fails when features exhibit strong multicollinearity. |
| **MLP** | Learns arbitrary non-linear representations. | Black-box nature; sensitive to convergence limits and tuning. |

---

### D. Is Accuracy Alone Sufficient for Evaluation?

**No, accuracy alone is insufficient for evaluating medical classification tasks.**

1. **Impact of Class Imbalance:** Accuracy treats all errors equally. A trivial model predicting "Non-Diabetic" for every sample would still yield high accuracy while failing entirely to catch diseased individuals.
2. **Clinical Costs of False Negatives:** A False Negative (failing to diagnose a diabetic patient) can delay medical intervention, leading to severe health complications. Thus, **Recall (Sensitivity)** and **ROC-AUC** are significantly more important clinical benchmarks than overall Accuracy.