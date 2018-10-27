import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Dense

from assignment1 import read_data_set

# read the data set and divide it into feature(x) and target(y)
data_set = read_data_set()
X = data_set.iloc[:, 0:15].values
y = data_set.iloc[:, 15].values


# ----------------------------------Preprocessing--------------------------------------------------- #

# encode the label data (i.e. non-numerical data) to numerical data
labelencoder_A1 = LabelEncoder()
X[:, 0] = labelencoder_A1.fit_transform(X[:, 0])
labelencoder_A4 = LabelEncoder()
X[:, 3] = labelencoder_A4.fit_transform(X[:, 3])
labelencoder_A5 = LabelEncoder()
X[:, 4] = labelencoder_A5.fit_transform(X[:, 4])
labelencoder_A6 = LabelEncoder()
X[:, 5] = labelencoder_A6.fit_transform(X[:, 5])
labelencoder_A7 = LabelEncoder()
X[:, 6] = labelencoder_A7.fit_transform(X[:, 6])
labelencoder_A9 = LabelEncoder()
X[:, 8] = labelencoder_A9.fit_transform(X[:, 8])
labelencoder_A10 = LabelEncoder()
X[:, 9] = labelencoder_A10.fit_transform(X[:, 9])
labelencoder_A12 = LabelEncoder()
X[:, 11] = labelencoder_A12.fit_transform(X[:, 11])
labelencoder_A13 = LabelEncoder()
X[:, 12] = labelencoder_A13.fit_transform(X[:, 12])
labelencoder_A14 = LabelEncoder()
X[:, 13] = labelencoder_A14.fit_transform(X[:, 13])

# Encode categorical integer features as a one-hot numeric array.
# Dummy variable approach.
# onehotencoder = OneHotEncoder(categorical_features=[0, 3, 4, 5, 6, 8, 9, 11, 12, 13])
# X = onehotencoder.fit_transform(X).toarray()

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# ----------------------------------ANN Modeling--------------------------------------------------- #

# Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
input_nodes_count = X.shape[1]
classifier.add(Dense(output_dim=input_nodes_count, init='uniform', activation='relu', input_dim=input_nodes_count))

# Adding the second hidden layer
classifier.add(Dense(output_dim=14, init='uniform', activation='relu'))

# Adding the second hidden layer
# classifier.add(Dense(output_dim=5, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='relu'))

# Compiling Neural Network
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ------------------------------Training and Evaluation----------------------------------------------- #

# split data into train-set and test-set
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
# kfold = StratifiedKFold(n_splits=5, shuffle=True)

f1_measures = []
for train_set, test_set in kfold.split(X, y):
    # fitting our model
    classifier.fit(X[train_set], y[train_set], batch_size=15, epochs=50, verbose=0)

    # evaluate the model
    # 1. predicting the test set results
    y_pred = classifier.predict(X[test_set])
    y_pred = (y_pred > 0.5).astype(int)
    # 2. calculate f1 measure: F1 = 2 * (precision * recall) / (precision + recall)
    f1_measures.append(f1_score(y_true=y[test_set], y_pred=y_pred))

print(np.mean(f1_measures))
