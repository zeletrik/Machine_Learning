# Importing the libraries for preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

path_to_dataset = 'PATH TO DATASET'
test_size = 0.2

# Importing the dataset
dataset = pd.read_csv(path_to_dataset)

# Last column for Y, may change the index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into a traning and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)

# Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# Scale without dummy varaibles
# X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
# X_test[:, 3:]  = sc_X.transform(X_test[:, 3:])

# Scale dummy varaibles too
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""