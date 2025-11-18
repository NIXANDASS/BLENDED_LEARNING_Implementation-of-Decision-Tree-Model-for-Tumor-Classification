# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data Import the dataset to initiate the analysis.

2.Explore Data Examine the dataset to identify patterns, distributions, and relationships.

3.Select Features Determine the most important features to enhance model accuracy and efficiency.

4.Split Data Separate the dataset into training and testing sets for effective validation.

5.Train Model Use the training data to build and train the model.

6.Evaluate Model Measure the model’s performance on the test data with relevant metrics.

## Program:
```

Program to  implement a Decision Tree model for tumor classification.
Developed by: Nixan Dass A
RegisterNumber:  212222040109

```
```py
#Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Data Loading

data = pd.read_csv('tumor.csv')

#Step 2: Data Exploration
#Display the first few rows and column names for verification
print(data.head())
print(data.columns)

#Step 3: Select features and target variable

#Drop id and other non-feature columns, using diagnosis as the target
x = data.drop(columns=['Class']) # Remove any irrelevant columns
y = data['Class'] # The target column indicating benign or malignant diagnosis

#Step 4: Data Splitting

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Step 5: Model Training
#Initialize and train the Decision Tree Classifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Step 6: Model Evaluation
#Predicting on the test set

y_pred = model.predict(X_test)

#Calculate accuracy and print classification metrics

accuracy = accuracy_score(y_test, y_pred)
print("NIXAN DASS A")
print("212222040109")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlOrRd")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```


## Output:

<img width="1355" height="357" alt="image" src="https://github.com/user-attachments/assets/cd93de18-0134-450b-b93a-e13ffb0c160b" />
<img width="1354" height="94" alt="image" src="https://github.com/user-attachments/assets/355090fb-b826-4e89-b7d1-22f0c64199f6" />
<img width="1048" height="271" alt="image" src="https://github.com/user-attachments/assets/94017788-0e32-4a8f-8a9b-adb0b5ae69dd" />
<img width="1375" height="573" alt="image" src="https://github.com/user-attachments/assets/df1d9d58-33d8-4b00-9284-90f8855f95cc" />




## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
