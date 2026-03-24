import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Initialize models and scaler
scaler = StandardScaler()
rfc = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=42)
svc = SVC(kernel='linear')

# Load and inspect data
data = pd.read_csv('Dataset/dataset.csv')
print("Initial data shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nColumns:", data.columns)
print("\nCOPD counts:")
print(data['copd'].value_counts())
print("\nCOPD severity counts:")
print(data['COPDSEVERITY'].value_counts())

# Drop unnecessary columns
columns = ['Unnamed: 0', 'ID', 'COPDSEVERITY', 'MWT1', 'MWT2']
data.drop(columns=columns, axis=1, inplace=True)
print("\nData shape after dropping columns:", data.shape)

# Data cleaning
data.drop(data[data['AGE'] == 10].index, axis=0, inplace=True)
data.drop(data[data['AGE'] == 30].index, axis=0, inplace=True)
data.drop(data[data['copd'] == 30].index, axis=0, inplace=True)
data.drop(data[data['copd'] == 10].index, axis=0, inplace=True)

# Handle missing values
print("\nMissing values before imputation:")
print(data.isna().sum())
data.fillna(data.mean(), inplace=True)
print("\nMissing values after imputation:")
print(data.isna().sum())

# EDA Visualizations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='rocket', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Age distribution analysis
age = pd.DataFrame(data['AGE'].value_counts(bins=3)).reset_index()
age.columns = ['Range', 'Age']
age['Range'] = age['Range'].astype(str)
for i in range(age.shape[0]):
    age['Range'][i] = age['Range'].iloc[i][1:-1]
age['Range'][2] = '43.9551, 58.667'

plt.figure(figsize=(6, 5))
sns.barplot(data=age, x='Range', y='Age', palette='viridis')
plt.title('Age Distribution')
plt.xticks(rotation=0)
plt.show()

# COPD counts by age
df = data.groupby([pd.cut(data['AGE'], bins=3), 'copd']).size().unstack().reset_index().rename(columns={'index': 'AGE'})
df_melted = pd.melt(df, id_vars=['AGE'], var_name='COPD', value_name='Count')
plt.figure(figsize=(8, 6))
sns.barplot(x='AGE', y='Count', hue='COPD', data=df_melted, palette='viridis')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('COPD Counts by Age Range')
plt.legend(title='COPD')
plt.show()

# Smoking analysis
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
for count, i in enumerate(data['smoking'].unique()):
    ax[count].set_title(f'Smoking: {i}')
    ax[count].plot(data[data['smoking'] == i]['PackHistory'])
plt.tight_layout()
plt.show()

# MWT1Best by age
columns = ['AGE', 'MWT1Best']
df = pd.DataFrame({col: data[col] for col in columns})
df['Age Range'] = pd.cut(df['AGE'], bins=3)
df['MWT1Best'] = pd.to_numeric(df['MWT1Best'], errors='coerce')
df['Age Range'] = df['Age Range'].astype(str)
for i in range(df.shape[0]):
    df['Age Range'][i] = df['Age Range'].iloc[i][1:-1]

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Age Range', y='MWT1Best', palette='viridis')
plt.title('MWT1Best by Age Range')
plt.show()

# FVC and Diabetes analysis
df = data[['FVC', 'Diabetes']]
df['FVC Range'] = pd.cut(df.FVC, bins=3)
df = df.groupby(['FVC Range', 'Diabetes']).size().unstack(fill_value=0).reset_index()
df_melted = pd.melt(df, id_vars=['FVC Range'], var_name='Diabetes', value_name='Count')
df_melted['FVC Range'] = df_melted['FVC Range'].astype(str)
for i in range(df_melted.shape[0]):
    df_melted['FVC Range'][i] = df_melted['FVC Range'].iloc[i][1:-1]

plt.figure(figsize=(8, 6))
sns.barplot(data=df_melted, x='FVC Range', y='Count', hue='Diabetes', palette='viridis')
plt.title('Diabetes Count by FVC Range')
plt.show()

# Prepare data for modeling
X = data.loc[:, data.columns != 'copd']
y = data['copd']

# Check and map labels to 0-based indexing
print("\nOriginal label values:", np.unique(y))
label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
y_mapped = y.map(label_mapping)
print("Mapped label values:", np.unique(y_mapped))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)
print("\nTrain/test shapes:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# Prepare CNN input
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Build CNN model
num_classes = len(np.unique(y_mapped))
model = models.Sequential([
    layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(16, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train_cnn, y_train, epochs=200, batch_size=16, validation_split=0.1)
model.save('copd_cnn_model.h5')

# Training history visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(acc, label='Training Accuracy', color='orange')
ax[0].plot(val_acc, label='Validation Accuracy', color='blue')
ax[0].set_title('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(loss, label='Training Loss')
ax[1].plot(val_loss, label='Validation Loss')
ax[1].set_title('Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.show()

# Evaluate model
y_pred = model.predict(X_train_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_train, y_pred_classes))

print("\nAccuracy:", accuracy_score(y_train, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_train, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Optional: Reverse label mapping for interpretation
reverse_mapping = {v: k for k, v in label_mapping.items()}
y_test_original = np.vectorize(reverse_mapping.get)(y_train)
y_pred_original = np.vectorize(reverse_mapping.get)(y_pred_classes)

print("\nOriginal label classification report:")
print(classification_report(y_test_original, y_pred_original))