import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("intro_extro.csv")
df

features = df.drop('id', axis = 1).columns.tolist()

sns.set_palette('Set2')
plt.figure(figsize=(15, 10))

# Loop through and plot each histogram
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    ax = sns.countplot(x=col, data=df)
    
    plt.xlabel(col)
    plt.ylabel(' ')
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.show()


# Impute categorical (object/dtype) columns with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Impute continuous (numerical) columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# Split
X = df.drop(["id", "Personality"], axis=1)
y = df["Personality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
lgb_model = LGBMClassifier(
    n_estimators=98, max_depth=12, learning_rate=0.1,
    reg_lambda=1.15, subsample=0.95, colsample_bytree=1, random_state=43
)
lgb_model.fit(X_train, y_train)

# Convert X to float numpy array (ensure shape compatibility)
X_np = X_train.astype(np.float32).to_numpy()


# Define the input type
initial_type = [('input', FloatTensorType([None, X_np.shape[1]]))]

# Convert LightGBM model to ONNX
onnx_model = onnxmltools.convert_lightgbm(lgb_model, initial_types=initial_type)

# Save to file
with open("lgb_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())



