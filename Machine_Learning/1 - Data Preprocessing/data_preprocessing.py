# Importing the libraries
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('PATH_TO_DATA_CSV')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Takeing care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into a traning and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
sc_X = StandardScaler()

# Scale without dummy varaibles
# X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
# X_test[:, 3:]  = sc_X.transform(X_test[:, 3:])

# Scale dummy varaibles too
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)