import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# Read CSV data into pandas dataframe
data = pd.read_csv("TeamA_Results.csv")

# Get ERP Components
col_names = data['Component'].unique()

# Create column for each ERP component
data_n = pd.get_dummies(data['Component'])
test = pd.DataFrame(data_n.to_numpy() * data['Value'].to_numpy().reshape(-1, 1), columns=col_names)

# Merge the column wise component with original dataset and drop unnecessary columns
train = pd.concat([data, test], axis=1)
train = train.drop(['Frame', 'ID', 'Component', 'Value'], axis=1)

# represent gender as number
gender_dict = {'male': 1, 'female': 0}
train['Sex'] = train['Sex'].map(gender_dict)

# Get rid of all NA values
train = train.dropna()

# Separate dependent and independent variables
train_y = train['Sex']
train_x = train.drop(['Sex'], axis=1)

# Split the dataset into train and test

# Define random forest parameters to explore and tune
n_estimators_list = [2000, 5000, 10000, 20000]
max_depth_list = [100, 500, 1000]

parameters = {'n_estimators': n_estimators_list, 'max_depth': max_depth_list}

# Create random forest classifier instance
rf_classifier = RandomForestClassifier()

# Initialize gridsearch
clf = GridSearchCV(rf_classifier, parameters, verbose=3, n_jobs=5, refit='f1',
                   scoring=['f1', 'precision', 'recall'])

# Apply gridsearch to identify parameters
clf.fit(train_x, train_y)

print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
means_f1 = clf.cv_results_['mean_test_f1']
means_precision = clf.cv_results_['mean_test_precision']
means_recall = clf.cv_results_['mean_test_recall']
params = clf.cv_results_['params']
for f1, precision, recall,  param in zip(means_f1, means_precision, means_recall, params):
    print("%f (%f) [%f] with: %r" % (f1, precision, recall, param))

print("Done")

