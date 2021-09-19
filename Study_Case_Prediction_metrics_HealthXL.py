#Importing data
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def show_dtypes(base):
    print(base.dtypes)
    return None

def show_dimensions(base):
    # base dimensions
    print ('NUmber of Rows: {}'.format(base.shape[0]))
    print ('NUmber of Columns: {}'.format(base.shape[1],end='\n\n'))
    return None

def show_missing_values(base):
    print(pd.isnull(base).sum())
    return None

# load dataset
base = pd.read_csv('C:/spyder python/study case/Data Analyst CAse Study_May 2021.csv')

# base dimensions
show_dimensions(base)

# base type
show_dtypes(base)

# missing_values
show_missing_values(base)

#Filling up missing values
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
base['Current Type'] = imputer.fit_transform(base[['Current Type']])

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
base['Current Nurturing'] = imputer.fit_transform(base[['Current Nurturing']])

#Checking the quatity of each value
base['Date'].unique().shape
base['Name'].unique().shape
base['Current Type'].unique().shape
base['Current Nurturing'].unique().shape
base['Event'].unique().shape

#excluding attributes
base.drop(['Description'],1,inplace=True)
base.drop(['Date'],1,inplace=True)

show_dimensions(base)

show_dtypes(base)

show_missing_values(base)

# choosing label
base = base[['Name','Current Type','Current Nurturing','Event']]

#Spliting data in features and label
features= base.iloc[:,0:3].values
labels=base.iloc[:,3].values


#Transforming categorical variables in numeric variables, binarize data
labelencoder_features = LabelEncoder()

features[:,0] = labelencoder_features.fit_transform(features[:,0])
features[:,1] = labelencoder_features.fit_transform(features[:,1])
features[:,2] = labelencoder_features.fit_transform(features[:,2])

labelencorder_labels = LabelEncoder()
labels = labelencorder_labels.fit_transform(labels)

####Rescaling variables (transforming data in normal distribution)
scaler = StandardScaler()
features = scaler.fit_transform(features)


#Sharing data in train and test data 
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=0)

# Applaying ml algorithym desicion tree to make predictions

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(features_train, labels_train)
predictions = classifier.predict(features_test)


# Verfying metrics
acc = accuracy_score(labels_test, predictions)
matriz = confusion_matrix(labels_test, predictions)
classification_report = classification_report(labels_test, predictions)

# Show results
print(predictions)
print(acc)
print(matriz)
print(classification_report)
