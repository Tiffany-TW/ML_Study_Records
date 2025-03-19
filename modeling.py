"""
This script is used to train classifiers with different algorithms
"""

#%%
# import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer

# # self_defined package: core_lib
# import os
# import sys
# sys.path.append(os.path.join(os.getcwd(), "../../core_lib"))
# import packages.data_quality.data_quality as dq

#%%
# Additional functions
def check_missing(data:pd.DataFrame):
    col_name = []
    result = []
    ratio = []
    dashbd = pd.DataFrame()
    for col in data.columns:
        col_name.append(col)
        result.append(data[col].isna().any())
        ratio.append(data[col].isna().sum()/data[col].size)
    dashbd["Feature"] = col_name
    dashbd["Missing or not"] = result
    dashbd["Ratio"] = ratio
    dashbd.set_index("Feature")
    return dashbd

def check_attribute(data:pd.DataFrame):
    col_name = []
    result = []
    dashbd = pd.DataFrame()
    num_lst = data.select_dtypes(include='number').columns.to_list() # names for columns (numerical)
    str_lst = data.select_dtypes(include='object').columns.to_list() # names for columns (str)
    for col in data.columns:
        col_name.append(col)
        if col in num_lst:
            result.append("Numeric")
        else:
            result.append("Text")
    dashbd["Feature"] = col_name
    dashbd["Attribute"] = result
    dashbd.set_index("Feature")
    return dashbd

#%%
# load iris.csv data, including features and labels 
df = pd.read_csv("iris.csv")

#%%
# Step0: examine data quality using self_defined package (core_lib)
# report = dq.DataQuality(df) # fail due to no time_stamp in the data, but time_stamp is not necessary -> fix core_lib
# report.show_missing()
dashbd_missing = check_missing(df) # In this case, there is no need to drop or impute  missing value 
dashbd_attr = check_attribute(df) # In this case, there is no need to handle text/categrotical attributes
dashbd = dashbd_missing.merge(dashbd_attr, how="left", on=["Feature"])
print(dashbd)

#%%
# Step1: split training and test set using train_test_split

## paramter "stratify" is considered when representative feature(category) is known beforehand
strat_train_set, strat_test_set = train_test_split(df, test_size=0.2, random_state=42)

#%%
# Step2: prepare the data for machine learning algorithms

## seperate the predictors and the labels \ creates a copy of the data and does not affect strat_train_set
iris = strat_train_set.drop("Species", axis=1)
iris_label = strat_train_set["Species"].copy()

## clean the data
#1 drop ID column, deal with columns having missing values
#2 feature scaling and transformation
### make a pipeline to apply the same data preprocessing methods to test data
### make_pipeline() applied assigned transformers on the same dataset not in a stepwise order
### ColumnTransformer allows us to applied different transformer pipelines to different items
drop_item = ["Id"]
num_attributes = list(dashbd_attr[dashbd_attr["Attribute"] == "Numeric"]["Feature"])
for i in range(len(drop_item)):
    num_attributes.remove(drop_item[i]) # remove drop_items in numeric_attributes
num_pipeline = make_pipeline(MinMaxScaler(feature_range=(0,1)))
preprocessing = ColumnTransformer([("scale", num_pipeline, num_attributes), ("drop", "drop", drop_item)])
iris_preprocessed = preprocessing.fit_transform(iris)
print(iris_preprocessed)

#%%
clf = make_pipeline(preprocessing, DecisionTreeClassifier(random_state=42))
## cross validation
cv_results_tree = cross_validate(clf, iris, iris_label, cv=2, scoring = 'accuracy', return_estimator =True)
print(f"Decision Tree:{cv_results_tree['test_score']}") # mean accuracy for classifier
# %%
