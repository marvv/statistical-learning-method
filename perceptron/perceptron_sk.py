from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from perceptron import load_data
import pandas as pd
# df = pd.read_csv('mnist_train.csv', header=0)
# X = df
# print(type(X))
# print(X.shape)
# print(X.head())
train_X, train_y = load_data('mnist_train.csv')
test_X, test_y = load_data('mnist_test.csv')

classifier = Perceptron()
classifier.fit(train_X, train_y)
pred = classifier.predict(test_X)
acc = accuracy_score(test_y, pred)
print('Perceptron accuracy is: ' + str(acc))


