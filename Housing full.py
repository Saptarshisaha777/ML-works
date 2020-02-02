# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:12:01 2019

@author: Saptarshi Saha
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Saptarshi Saha\PycharmProjects\MyCAPTAIIN\venv\Lib\dataset\housing.csv')
print(dataset.head())
print(dataset.info())
dataset.hist(bins=50, figsize=(20, 15))
plt.show()

# creating train test data
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(dataset, test_size=.2, random_state=42)

# Making a new category to convert median income into income category of five groups 1,2,3,4,5

dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)  # ceil gets the smallest interger value
dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)

print("The new dataset plot \n\n")
dataset.hist(bins=50, figsize=(20, 15))
plt.show()

# Making a normal test set for comparison with stratified test set
train_set, test_set = train_test_split(dataset, test_size=.2, random_state=42)

# Now doing stratified sampling on the income category
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]

print(dataset["income_cat"].value_counts() / len(dataset))
print("For the non stratified test set\n", test_set["income_cat"].value_counts() / len(test_set),
      "\n For the Stratified test set\n", strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# DROPPING income_cat and returning the data to its original form
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print(strat_train_set.info())

########################### WORKING WITH TRAIN DATASET ########################

dataset = strat_train_set.copy()

#### VISUALIZING DATA ####
# 1.
dataset.plot(kind="scatter", x="longitude", y="latitude")
plt.show()
# 2
dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
# 3
dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=dataset["population"] / 100, label="population",
             figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

##CALCULATING STANDERED CORRELATION
corr_matrix = dataset.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

##CORRELATION WITH PANDAS
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(dataset[attributes], figsize=(12, 8))

##MAKING NEW DATA ATTRIBUTES
dataset["rooms_per_household"] = dataset["total_rooms"] / dataset["households"]
dataset["bedrooms_per_room"] = dataset["total_bedrooms"] / dataset["total_rooms"]
dataset["population_per_household"] = dataset["population"] / dataset["households"]

# CALCULATION OF NEW CORRELATION
corr_matrix = dataset.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

####################  PREPARING DATA FOR ML ALGOS  #############################

dataset = strat_train_set.drop("median_house_value",
                               axis=1)  # "DROP" CREATES A COPY OF DATA WITHOUT EFFECTING THE ORIGINAL ONE
dataset_labels = strat_train_set["median_house_value"].copy()

##################### D A T A     C L E A N I N G ##############################

# REMOVING MISSING FEATURES (Using General function )

# Option 1 (Getting Rid of Corresponding Districts)
dataset.dropna(subset=["total_bedrooms"])

# Option 2 (Getting Rid of the whole Attribute)
dataset.drop("total_bedrooms", axis=1)

# Option 3 (Setting Values to some value like zero,Median,mean,etc)
median = dataset["total_bedrooms"].median()
dataset["total_bedrooms"].fillna(median, inplace=True)

print(dataset.info())

##### REMOVING MISSING FEATURES (Using Scikit learn) #######
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

dataset_num = dataset.drop("ocean_proximity",
                           axis=1)  # REMOVING ocean proximity data as it is an object data and thus MEDIAN cannot be calculated
imputer.fit(dataset_num)  # IMPUTER "REPLACES" all the missing values of each of the attributes with the median

print(imputer.statistics_)
X = imputer.transform(dataset_num)
dataset_tr = pd.DataFrame(X, columns=dataset_num.columns)

### HANDLING TEXT AND CATAGORICAL ATTRIBUTES
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
dataset_cat = dataset["ocean_proximity"]
dataset_cat_encoded = encoder.fit_transform(dataset_cat)

print(dataset_cat_encoded, encoder.classes_)

# INTRODUCING OneHotEncoder TO DECREASE THE CHANCE OF COMPARISION FAULT OR PRIORITY FAULT
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
dataset_cat_1hot = encoder.fit_transform(dataset_cat_encoded.reshape(-1, 1))

print(dataset_cat_1hot)  # Shows a Scipy sparse matrix
print(dataset_cat_1hot.toarray())

## WE CAN USE LabelBinarizer TO CONVERT TEXT TO INTEGER THEN INTEGER TO ONE-HOT VECTOR #############
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
dataset_cat_1hot = encoder.fit_transform(dataset_cat)

print(
    dataset_cat_1hot)  # This returns a dense NumPy array by default. We can get a sparse matrix instead by passing (sparse_output=True) to the LabelBinarizer

## CUSTOM TRANSFORMER
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, rooms_ix]
        population_per_household = X[:, bedrooms_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
dataset_extra_attribs = attr_adder.transform(dataset.values)

######################## TRANSFORMATION PIPELINES ##############################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

## D E M O ##
# num_pipeline = Pipeline([('imputer',SimpleImputer(strategy = "median" ))
#                            ,('attribs_adder',CombinedAttributesAdder())
#                           ,('std_scaler',StandardScaler())])

# dataset_num_tr =  num_pipeline.fit_transform(dataset_num)

## MAKING A CUSTOM TRANSFORMER FOR CONVERTING A PANDAS DATAFRAME for DIRECT use in PIPELINE ##

from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


##### MAKING A CUSTOM ### LabelBinarizer ### TRANSFORMER since it doesnt allow MORE THAN 2 POSITIONAL ARGUMENTS for DIRECT use in PIPELINE ##
class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        encoder = LabelBinarizer(sparse_output=self.sparse_output)
        return encoder.fit_transform(X)


###### FINAL PIPELINE APPLICATION ########

num_attribs = list(dataset_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler())])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('label_binarizer', MyLabelBinarizer())])

###### CONCATENATING 2 DIFF PIPELINES #######

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline), ("cat_pipeline", cat_pipeline)])

dataset_prepared = full_pipeline.fit_transform(dataset)

##### TRAINING AND EVALUATING ON THE TRAINING SET ###

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(dataset_prepared, dataset_labels)  ## OUR TRAINING MODEL IS PREPARED

model = LinearRegression()
model.fit(dataset_prepared, dataset_labels)

# some_data = dataset.iloc[:5]
# some_labels = dataset_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# p = lin_reg.predict(some_data_prepared)

x_check = dataset_prepared[:5]
y_check = dataset_labels.iloc[:5]
# x_prep=full_pipeline.fit_transform(x_check)
# x_check_tr =  num_pipeline.fit_transform(x_check)
# x_checkchar_tr =  cat_pipeline.fit_transform(x_check)
p = model.predict(x_check)









