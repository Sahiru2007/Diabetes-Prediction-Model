

# Diabetes Prediction Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to predict diabetes. The model uses various classifiers and provides evaluation metrics to compare their performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [ROC Curve](#roc-curve)
- [Saving the Model](#saving-the-model)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Overview

The notebook builds and evaluates several machine learning models to predict diabetes. The steps include:

1. Loading and exploring the dataset.
2. Data preprocessing.
3. Splitting the data into training and testing sets.
4. Building multiple classifiers.
5. Evaluating the models using various metrics.
6. Plotting the ROC Curve.
7. Saving the best model.

## Dataset

The dataset used in this notebook is the Pima Indians Diabetes Database, which can be found [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). This dataset contains several medical predictor variables and one target variable, `Outcome`.

### Dataset Description

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1)

## Installation

To run this notebook, you need to have Python installed along with the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository:

```sh
git clone <repository-url>
cd <repository-folder>
```

2. Open the Jupyter Notebook:

```sh
jupyter notebook Diabetes_prediction.ipynb
```

3. Run all cells in the notebook to see the complete analysis and model evaluation.

## Data Preprocessing

### Handling Missing Values

Missing values in the dataset are handled by replacing them with the mean of the respective columns:

```python
data = data.fillna(data.mean())
```

### Feature Scaling

Standardization of features is done to bring all features to a similar scale:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('Outcome', axis=1))
```

### Splitting the Data

The dataset is split into training and testing sets using a 70-30 split:

```python
from sklearn.model_selection import train_test_split

X = data_scaled
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Model Building

### Models Used

The notebook evaluates several models:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**

### Training the Models

Example: Training a Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Evaluation

### Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive instances to the total predicted positives.
- **Recall**: The ratio of true positive instances to the actual positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A summary of prediction results on a classification problem.

### Example: Evaluating Random Forest Classifier

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{cm}')
```

### Evaluation Results

- **Logistic Regression**: Accuracy ~ 75%
- **KNN**: Accuracy ~ 74%
- **Decision Tree**: Accuracy ~ 72%
- **Random Forest**: Accuracy ~ 77%
- **Gradient Boosting**: Accuracy ~ 76%

## ROC Curve

The ROC Curve is plotted to evaluate the performance of the classifiers. The ROC curve shows the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate).

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, color='blue', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc}')
```

### ROC Curve Plot

The ROC curve helps in understanding the performance of the model. A model with a higher area under the ROC curve (AUC) is considered better. For instance, the Random Forest model achieved an AUC of around 0.85.

<img width="711" alt="Screenshot 2024-06-24 at 21 40 04" src="https://github.com/Sahiru2007/Diabetes-Prediction-Model/assets/75121314/b1ee74c7-e4c5-40e5-a73b-46dc2d17094e">

## Saving the Model

The best-performing model is saved using the `pickle` module for future use:

```python
import pickle

filename = 'diabetes_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {filename}")
```

## Screenshots

Including screenshots of the notebook's output helps in understanding the workflow and results. Here are some examples:

### Data Distribution
Histograms of each feature to show data distribution:
<img width="728" alt="Screenshot 2024-06-24 at 21 37 48" src="https://github.com/Sahiru2007/Diabetes-Prediction-Model/assets/75121314/0bf42994-0d82-48eb-a776-e2e4ec7f18d7">

### Correlation Matrix
Heatmap showing correlations between features:
<img width="930" alt="Screenshot 2024-06-24 at 21 38 28" src="https://github.com/Sahiru2007/Diabetes-Prediction-Model/assets/75121314/11f0cfa7-1bad-401d-bba5-88fc2f21108d">

### Model Accuracy Comparison
Bar plot comparing the accuracy of different models:
<img width="1097" alt="Screenshot 2024-06-24 at 21 39 06" src="https://github.com/Sahiru2007/Diabetes-Prediction-Model/assets/75121314/e760c431-9406-4858-8496-dee03e5a1aab">

### ROC Curve
ROC curve for the best-performing model:
<img width="711" alt="Screenshot 2024-06-24 at 21 40 04" src="https://github.com/Sahiru2007/Diabetes-Prediction-Model/assets/75121314/b1ee74c7-e4c5-40e5-a73b-46dc2d17094e">

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

