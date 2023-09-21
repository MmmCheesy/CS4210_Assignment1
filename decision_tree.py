#-------------------------------------------------------------------------
# AUTHOR: Alexander Eckert
# FILENAME: decision_tree.py
# SPECIFICATION: ID3 ML Algorithm
# FOR: CS 4210- Assignment #1
# TIME SPENT: 0.5 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

# Mapping for feature transformation
age_mapping = {'Young': 1, 'Presbyopic': 3, 'Prepresbyopic': 2}
prescription_mapping = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_mapping = {'Yes': 1, 'No': 2}
tear_production_mapping = {'Reduced': 1, 'Normal': 2}

# Mapping for class transformation
lenses_mapping = {'Yes': 1, 'No': 2}

for row in db:
    # Transforming features
    features = [
        age_mapping[row[0]],
        prescription_mapping[row[1]],
        astigmatism_mapping[row[2]],
        tear_production_mapping[row[3]]
    ]
    X.append(features)

    # Transforming classes
    Y.append(lenses_mapping[row[4]])

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
