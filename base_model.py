import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pickle

wine_dataset = load_wine()

#loading the data to pandas dataset

all_data = wine_dataset['data']
features = wine_dataset['feature_names']
target = wine_dataset['target']
df = pd.DataFrame(data = all_data, columns = features)
df['target'] = target

# splitting the dataset to training and testing
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(df.drop('target',axis = 1), target,
                                                    test_size=test_size, random_state=0)

# building a classifier using descision tree classifier
md = DecisionTreeClassifier().fit(x_train, y_train)
print("accuracy is {0}".format(md.score(x_test, y_test)))


# need to save the model using pickle to the file
file_destination = "model/classifier_dump.sav"
pickle.dump(md,open(file_destination, 'wb'))

# testing the dumped model by loading the model
model_loaded = pickle.load(open(file_destination, 'rb'))
accuracy = model_loaded.score(x_test, y_test)
print(accuracy)
# the accuracy matches

