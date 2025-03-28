## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.Generate Confusion Matrix

## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: Keerthika.S
RegisterNumber:  212223040093
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
![424376687-40fa9158-e227-460b-a8cb-24b6888d284b](https://github.com/user-attachments/assets/34686244-d89e-4c7f-a964-a3639080c701)

![424376818-a862cdb7-58fe-4f3c-86cb-9284559715e0](https://github.com/user-attachments/assets/a3a351f1-990e-4626-a71c-39d00cb4e5cb)

![424376879-e6529585-3244-4255-a2cf-8dec45a02cc8](https://github.com/user-attachments/assets/70960534-c9c0-439c-9150-568bdbc3f532)

![424376978-3c0567b7-dc50-48e2-b42b-36215839d63f](https://github.com/user-attachments/assets/27b7a164-c4f6-4041-91bb-31acb952c46c)

![424378269-27b76a7c-2f6a-4bbb-9f04-091d4f6bc886](https://github.com/user-attachments/assets/df1fc1bf-4714-4596-89af-e2fdb714bc56)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
