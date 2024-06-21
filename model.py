import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the data from the Excel file
data = pd.read_excel('dataset.xls')

# Define the mappings for categorical variables
mapping_pendapatan = {'<200000': 0, '200000-400000': 1, '>400000': 2}
mapping_status = {'Menikah': 0, 'Belum Menikah': 1, 'Cerai': 2}
mapping_tempat_tinggal = {'Punya': 0, 'Dengan Orang Tua': 1, 'Kontrak': 2}
mapping_mobil = {'Punya': 0, 'Tidak Punya': 1}
mapping_lolos_persyaratan = {'Tidak Lolos': 0, 'Lolos': 1}

# Apply the mappings
data['Pendapatan'] = data['Pendapatan'].map(mapping_pendapatan)
data['Status'] = data['Status'].map(mapping_status)
data['Tempat Tinggal'] = data['Tempat Tinggal'].map(mapping_tempat_tinggal)
data['Mobil'] = data['Mobil'].map(mapping_mobil)
data['Lolos Persyaratan'] = data['Lolos Persyaratan'].map(mapping_lolos_persyaratan)

# Define the feature columns and the target column
feature = ['Pendapatan', 'Status', 'Anak', 'Tempat Tinggal', 'Mobil']
target = 'Lolos Persyaratan'

# Split the data into features (X) and target (y)
X = data[feature]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model
nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(nb, 'naive_bayes_model.pkl')

# Evaluate the model
y_pred = nb.predict(X_test)

# Print evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
