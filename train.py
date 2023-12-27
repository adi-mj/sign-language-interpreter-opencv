import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

df = pd.read_csv('df.csv')
df.columns = [i for i in range(df.shape[1])]

df = df.rename(columns={63: 'Output'})
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
svm = SVC(C=10, gamma=0.1, kernel='rbf')
svm.fit(x_train, y_train)

print(svm.score(x_test, y_test))

import pickle

# save model
with open('model2.pkl','wb') as f:
    pickle.dump(svm,f)