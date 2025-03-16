# ML_Study_Records
This repository contains notes about machine leaning, including concepts and codings for algorithms, evaluation metrics, etc. 
## Evaluation metrics
### Classification
For classification methods, the following metrics are commonly used to evaluate the performance of models:
1. Confusion matrices
2. Accuracy of the classifier
3. Precision of the classifier
4. Recall, also called true positive rate, of the classifier
5. F1 score
6. Precision\Recall (PR) curve
7. The ROC curve
#### Confusion matrices
A confusion matrix is a 2-dimensional n by n matrix. Each row in a confusion matrix represents an actual class, while each column represents a predicted class. Each entry indicates the count of instances of class A are classified as class B. Here, we take a two class classification problem (labeled with negative and positive) as an example. That is, n is equal to 2. Then, the confusion matrix is shown as the following:
|Actual/Prediction|Negative|Positive|
|----|----|----|
|**Negative**|True negative (TN)|False positive (FP)|
|**Positive**|False negative (FN)|True positive (TP)|

Note: When a binary classification problem is labeled with negative and positive, this tells us that the problem is discussed for a specific class.

After a confusion matrix has been computed, more concise metrics such as **accuracy**, **precision**, **recall**, and **the ROC curve** could be computed accordingly.
#### Accuracy of the classifier
Accu
#### Precision for a specific class of the classifier
For a given class, precision of the classifier is an indicator for the accuracy of the positive predictions, defined as $$\frac{TP}{TP+FP}$$ . In other words, precision is a measurement of how many positive predictions are exactly positive instances.

*Feature*:

*Pros*:

*Cons*:

*Sci-Kit learn implementation*:
#### Recall for a specific class of the classifier
For a given class, recall of the classifier is an indicator for the sensitivity of the positive predictions, defined as $$\frac{TP}{TP+FN}$$ . In other words, recall is a measurement of how many positive instances are correctly predicted as positive.

*Feature*:

*Pros*:

*Cons*:

*Sci-Kit learn implementation*:
#### F1 score
F1 score is a combination of precision and recall. It is defined as the harmonic mean of precision and recall, ie. $$\frac{2}{\frac{1}{precision}+\frac{1}{recall}}$$ .

#### Reference
 
