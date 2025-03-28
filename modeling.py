"""
This script is used to train classifiers with different algorithms
"""

#%%
# import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_curve

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
df = pd.read_csv("data/iris.csv")

#%%
# Step0: examine data quality using self_defined package (core_lib)
# report = dq.DataQuality(df) # fail due to no time_stamp in the data, but time_stamp is not necessary -> fix core_lib
# report.show_missing()
dashbd_missing = check_missing(df) # In this case, there is no need to drop or impute  missing value 
dashbd_attr = check_attribute(df) # In this case, there is no need to handle text/categrotical attributes
dashbd = dashbd_missing.merge(dashbd_attr, how="left", on=["Feature"])
# print(dashbd)

#%%
# Step1: split training and test set using train_test_split

## paramter "stratify" is considered when representative feature(category) is known beforehand
strat_train_set, strat_test_set = train_test_split(df, test_size=0.2, random_state=42)

#%%
# Step2: prepare the data for machine learning algorithms

## seperate the predictors and the labels \ creates a copy of the data and does not affect strat_train_set
### train
iris = strat_train_set.drop("Species", axis=1)
iris_label = strat_train_set["Species"].copy()
### test
iris_test = strat_test_set.drop("Species", axis=1)
iris_test_label = strat_test_set["Species"].copy()
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
# print(iris_preprocessed)

#%%
# Step3: modeling / try different classifiers and compare the performance metrics 

## Training a binary classifier (True: Iris-setosa; Iris-versicolor; Iris-virginica)
species = "Iris-virginica"
iris_train_label_bi = (iris_label == species)
iris_test_label_bi = (iris_test_label == species)

### decision tree
clf = make_pipeline(preprocessing, DecisionTreeClassifier(random_state=42))
cv_acc_tree = cross_val_score(clf, iris, iris_train_label_bi, cv=3, scoring = 'accuracy') # cross validation
cv_predict_tree = cross_val_predict(clf, iris, iris_train_label_bi, cv=3)
cv_score_tree = cross_val_predict(clf, iris, iris_train_label_bi, cv=3, method="predict_proba")
cm_tree = confusion_matrix(iris_train_label_bi, cv_predict_tree)
precisions_tree, recalls_tree, thresholds_tree = precision_recall_curve(iris_train_label_bi, cv_score_tree[:,1])

# precision-recall trade-off
# plt.plot(thresholds_tree, precisions_tree[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds_tree, recalls_tree[:-1], "g-", label = "Recall", linewidth=2)
# plt.vlines(0.01, 0, 1, "k", "dotted", label="threshold")
# plt.legend()
# plt.xlabel("Threshold")
# plt.show()

# print(cv_predict_tree)
# print(cv_score_tree)
print(f"Decision Tree accuracy:{cv_acc_tree}") # mean accuracy for classifier
print(f"Decision Tree CM:{cm_tree}")
print(f"Decision tree precision:{precision_score(iris_train_label_bi, cv_predict_tree)}")
print(f"Decision tree recall:{recall_score(iris_train_label_bi, cv_predict_tree)}")
print(f"Decision tree f1_score:{f1_score(iris_train_label_bi, cv_predict_tree)}")

### dummy classifier
dummy_clf = make_pipeline(preprocessing, DummyClassifier())
cv_acc_dummy = cross_val_score(dummy_clf, iris, iris_train_label_bi, cv=3, scoring = 'accuracy')
cv_predict_dummy = cross_val_predict(dummy_clf, iris, iris_train_label_bi, cv=3)
cm_dummy = confusion_matrix(iris_train_label_bi, cv_predict_dummy)
print(f"Dummy:{cv_acc_dummy}") # mean accuracy for classifier
print(f"Dummy CM:{cm_dummy}")
print(f"Dummy precision:{precision_score(iris_train_label_bi, cv_predict_dummy)}")
print(f"Dummy recall:{recall_score(iris_train_label_bi, cv_predict_dummy)}")
print(f"Dummy f1_score:{f1_score(iris_train_label_bi, cv_predict_dummy)}")

### SGDClassifier
sgd_clf = make_pipeline(preprocessing, SGDClassifier(random_state=42))
cv_acc_sgd = cross_val_score(sgd_clf, iris, iris_train_label_bi, cv=3, scoring="accuracy")
cv_predict_sgd = cross_val_predict(sgd_clf, iris, iris_train_label_bi, cv=3)
cv_score_sgd = cross_val_predict(sgd_clf, iris, iris_train_label_bi, cv=3, method="decision_function")
cm_sgd = confusion_matrix(iris_train_label_bi, cv_predict_sgd)
precisions_sgd, recalls_sgd, thresholds_sgd = precision_recall_curve(iris_train_label_bi, cv_score_sgd)
fpr_sgd, tpr_sgd, thresholds_sgd_roc = roc_curve(iris_train_label_bi, cv_score_sgd)

# precision-recall trade-off
# plt.plot(thresholds_sgd, precisions_sgd[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds_sgd, recalls_sgd[:-1], "g-", label = "Recall", linewidth=2)
# plt.vlines(10, 0, 1, "k", "dotted", label="threshold")
# plt.legend()
# plt.xlabel("Threshold")
# plt.show()

# print(cv_predict_sgd)
# print(cv_score_sgd)
print(f"SGD_Classifier:{cv_acc_sgd}") # mean accuracy for classifier
print(f"SGD Confusion matrix:{cm_sgd}")
print(f"SGD precision:{precision_score(iris_train_label_bi, cv_predict_sgd)}")
print(f"SGD recall:{recall_score(iris_train_label_bi, cv_predict_sgd)}")
print(f"SGD f1_score:{f1_score(iris_train_label_bi, cv_predict_sgd)}")


# PR curve between decision tree and sgd classifier
idx_for_90_precision = (precisions_sgd >= 0.9).argmax()
threshold_for_90_precision = thresholds_sgd[idx_for_90_precision]
print(threshold_for_90_precision)
# plt.plot(recalls_tree, precisions_tree, "b--", linewidth=2, label="Decision Tree")
# plt.plot(recalls_sgd, precisions_sgd, "b-", linewidth=2, label="SGD")
# plt.legend()
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Comparing PR curves between the decision tree classifier and the sgd classifier")
# plt.show()

# ROC curve
idx_for_thresh_at_90 = (thresholds_sgd_roc <= threshold_for_90_precision).argmax()
fpr_90, tpr_90 = fpr_sgd[idx_for_thresh_at_90], tpr_sgd[idx_for_thresh_at_90]
plt.plot(fpr_sgd, tpr_sgd, linewidth=2, label="ROC curve")
plt.plot([0,1], [0,1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label=f"Threshold for 90% precision = {round(thresholds_sgd_roc[idx_for_thresh_at_90], 3)}")
plt.legend()
plt.xlabel("False Positive Rate (Fall-out)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("The ROC curve of both the SGD classifier and the random classifier")
plt.show()

# select best threshold interval for the decision function of SGD classifier
metric_df = pd.DataFrame()
metric_df["precisions"] = precisions_sgd[:-1]
metric_df["recalls"] = recalls_sgd[:-1]
metric_df["thresholds"] = thresholds_sgd
print(metric_df)

select_df = metric_df[(metric_df["precisions"] <= metric_df["recalls"]) & (metric_df["precisions"] > 0.9)]
print(select_df)

# By adjusting the threshold of decision function, the binary classification result differs
thresh = -1.143001
idx_for_thresh = (cv_score_sgd >= thresh)
iris_label_decision = pd.DataFrame()
pred = []
iris_label_decision["correct answer"] = iris_train_label_bi
iris_label_decision["origin_pred"] = cv_predict_sgd
for i in range(len(iris)):
    if idx_for_thresh[i] == True:
        pred.append(True)
    else:
        pred.append(False)
iris_label_decision[f"threshold={thresh}"] = pred

print(iris_label_decision[iris_label_decision["correct answer"]!= iris_label_decision[f"threshold={thresh}"]])
print(iris_label_decision[iris_label_decision["correct answer"]!= iris_label_decision["origin_pred"]])
print(recall_score(iris_train_label_bi, iris_label_decision[f"threshold={thresh}"]))

# Apply the model to test set
sgd_clf.fit(iris, iris_train_label_bi)
final_predictions = sgd_clf.predict(iris_test)
final_recall = recall_score(iris_test_label_bi, final_predictions)
final_precision = precision_score(iris_test_label_bi, final_predictions)
final_decision = sgd_clf.decision_function(iris_test)
print(f"precision:{final_precision}")
print(final_recall)

idx_for_thresh_test = (final_decision >= thresh)
iris_label_decision_test = pd.DataFrame()
pred_test = []
iris_label_decision_test["correct answer"] = iris_test_label_bi
iris_label_decision_test["origin_pred"] = final_predictions
for i in range(len(iris_test)):
    if idx_for_thresh_test[i] == True:
        pred_test.append(True)
    else:
        pred_test.append(False)
iris_label_decision_test[f"threshold={thresh}"] = pred_test
print(iris_label_decision_test)
print(recall_score(iris_test_label_bi, iris_label_decision_test[f"threshold={thresh}"]))
print(precision_score(iris_test_label_bi, iris_label_decision_test[f"threshold={thresh}"]))
print(confusion_matrix(iris_test_label_bi, iris_label_decision_test[f"threshold={thresh}"]))
# %%