# ML_Study_Records
This repository contains notes about machine leaning, including concepts and codings for algorithms, evaluation metrics, etc. 
## Evaluation metrics
### Classification
For classification methods, the following metrics are commonly used to evaluate the performance of models:
1. [Confusion matrices](#confusion-matrices)
2. [Accuracy of the classifier](#accuracy-of-the-classifier)
3. [Precision of the classifier](#precision-for-a-specific-class-of-the-classifier)
4. [Recall, also called true positive rate, of the classifier](#recall-for-a-specific-class-of-the-classifier)
5. [F1 score](#f1-score)
6. [Precision\Recall (PR) curve](#precisionrecall-pr-curve)
7. [The Reciever Operating Characteristics (ROC) curve](#the-reciever-operating-characteristics-roc-curve)
#### Confusion matrices
A confusion matrix is a 2-dimensional n by n matrix. Each row in a confusion matrix represents an actual class, while each column represents a predicted class. Each entry indicates the count of instances of class A are classified as class B. Here, we take a two class classification problem (labeled with negative and positive) as an example. That is, n is equal to 2. Then, the confusion matrix is shown as the following:
|Actual/Prediction|Negative|Positive|
|----|----|----|
|**Negative**|True negative (TN)|False positive (FP)|
|**Positive**|False negative (FN)|True positive (TP)|

Note: When a binary classification problem is labeled with negative and positive, this tells us that the problem is discussed for a specific class.

After a confusion matrix has been computed, more concise metrics such as **accuracy**, **precision**, **recall**, and **the ROC curve** could be computed accordingly.
#### Accuracy of the classifier
Accuracy is an indicator for the correctedness of the total predictions. It is defined as $$\frac{TN+TP}{TN+TP+FN+FP}$$
#### Precision for a specific class of the classifier
For a given class, precision of the classifier is an indicator for the accuracy of the positive predictions, defined as $$\frac{TP}{TP+FP}$$ . In other words, precision is a measurement of how many positive predictions are exactly positive instances.

*Features*:
1. $ 0 \leq \text{precision} \leq 1 $
2. The greater the value, the better the performance of the classifier

*Sci-Kit learn implementation*: *precision_score(true_label, prediction)*
#### Recall for a specific class of the classifier
For a given class, recall of the classifier is an indicator for the sensitivity of the positive predictions, defined as $$\frac{TP}{TP+FN}$$ . In other words, recall is a measurement of how many positive instances are correctly predicted as positive, which is also called **sensitivity** or **TPR**.

*Features*:
1. $ 0 \leq \text{recall} \leq 1 $
2. The greater the value, the better the performance of the classifier

*Sci-Kit learn implementation*: *recall_score(true_label, prediction)*
#### F1 score
F1 score is a combination of precision and recall. It is defined as the harmonic mean of precision and recall, ie. $$\frac{2}{\frac{1}{precision}+\frac{1}{recall}}$$ .

*Features*:
1. $ 0 < \text{F1 score} \leq 1 $
2. The greater the value, the better the performance of the classifier
3. F1 score is commonly used to compare among several classifiers who has similar precision and recall.
4. Observe the following cases, there are several possible combinations of paired precision and recall that result in the same F1 score:    
    - When precision = 1 and recall = 0.5, then F1 score = $\frac{2}{3} \approx$ 0.66 
    - When precision = 0.6, and recall = 0.75, then F1 score = $\frac{2}{3} \approx$ 0.66
    - When precision = $\frac{2}{3}$, and recall = $\frac{2}{3}$, then F1 score = $\frac{2}{3} \approx$ 0.66
    - In summary, with similar precision and recall ([Suppl](#supplementary) [1,2]), the uniqueness of F1 score is guaranteed. Otherwise, comparison among different classifiers is meaningless.

*Sci-Kit learn implementation*: *f1_score(true_label, prediction)*

#### Precision/Recall (PR) Curve
Precision/Recall curve provides visualization for determining suitable thresholds for classifiers. Moreover, it shows apparent trends to illustrate the precision/recall trade-off.

*Features*:
1. The curve of a good classifier is closer to the top-right corner.
2. It is suitable for the cases where the positive class is rare or when we care more about the false positives than the false negatives. 

*Sci-Kit learn implementation*: *precision_recall_curve(true_label, prediction)*

#### The Reciever Operating Characteristics (ROC) curve
The ROC curve plots **recall** against **false positive rate (FPR)**. FPR is defined as $\frac{FP}{TP+FP}$ . Actually, FPR is also equal to 1 - TNR (true negative rate, called specificity). TNR is defined as $\frac{TN}{TN+FN}$ .

*Features*:
1. There is trade-off between recall (TPR) and FPR. That is, the higher the recall (TPR), the more FPR the classifier produces.
2. The ROC curve of a good classifier is closer to the top-left corner.
3. The area under the curve (AUC) of a ROC curve of a purely random classifier is 0.5. For a perfect classifier, its AUC is equal to 1.

*Sci-Kit learn implementation*: roc_curve(true_label, prediction)

#### Supplementary
[1] Having similar precision and recall is not always the case. Sometimes, we prefer precision over recall. Sometimes, we care more about recall. It depends on the contexts. Please read p.111 for examples.

[2] Because of the precision/recall trade-off, it is impossible for us to have high values both ways: Please read [Ref](#reference) [3] for more theoretical explaination.

#### Implementation using Iris dataset
Decision Tree classifier and SDG classifier are applied to train models for the classification problem of Iris flower.

**Binary classification problem** 
Description: The binary classification problem aims at detecting Iris-virginica through the length and width of sepals and petals.
- Skewed Dataset? : Iris-virginica comprises one-third of the total sample size. Even though the amount of positive and negative samples are not equal, it is not likely to be imbalanced. Please see more info on [Ref](#reference) [4].
- Training Results: 
    - Basic evaluation metrics of both Decision Tree classifier, and SDG classifier are shown in the following table. The table also includes dummy classifier as the control group. (cv = 3)

        ||Dummy|DT|SGD|
        |----|----|----|----|
        |Mean Accuracy|0.675|0.95|0.975|
        |Precision|ill-defined|0.923|0.974|
        |Recall|ill-defined|0.923|0.949|
        |F1 score|ill-defined|0.923|0.961|

    - Confusion matrices

        |Dummy|Negative|Positive|
        |----|----|----|
        |Negative|81|0|
        |Positive|39|0|

        |DT|Negative|Positive|
        |----|----|----|
        |Negative|78|3|
        |Positive|3|36|

        |SGD|Negative|Positive|
        |----|----|----|
        |Negative|80|1|
        |Positive|2|37|
    
    From the above metrics, we may conclude that the performance of the SGD classifier is better than the decision tree classifier. This conclusion can also be validated by the PR curve, where the curve of SGD is more closer to the top-right corner than that of the decision tree.
    - PR curve
    ![PR](/Figure/Figure_modeling_3.png)
    In addition to justifying the performance of the two classifiers, the PR curve also provides information for fine-tuning the classifier. In this case, I prefer better recall over precision since I expect to have all the Iris-virginica detected. It sounds great when precision and recall are 0.926829 and 0.974359 respectively. In this case, decision score equals to -1.143001. This value will be helpful for fine-tuning the SGD classifier.
    - ROC curve  
    ![ROC](/Figure/Figure_modeling_5.png)
    The ROC curve of the SGD classifier is toward the top-left of the figure. This again indicates that the SGD classifier is quite a good model for this binary classification problem.
* Fine-tuning the model and apply the classifer to the test set:
We are not allowed to modify the value of decision functions directly in the function. Instead, we calculate the decision score of the prediction and  classify true instances from false instances based on the assigned threshld. After applying the threshold which is equal to -1.143001, the evaluation metrics for the test set are shown as follows:
    * Confusion matrix

    ||Negative|Positive|
    |----|-----|-----|
    |Negative|18|1|
    |Positive|0|11|

    * precision and recall scores are 0.917 and 1 respectively.

    **To sum up, the classifier for the binary classification problem performs well according to the evaluation metrics on the test set.**






#### Reference
[1] https://www.evidentlyai.com/classification-metrics/multi-class-metrics

[2] Scikit-learn, L. W. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow. Ò Reilly Media.

[3] https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-4571(199401)45:1%3C12::AID-ASI2%3E3.0.CO;2-L

[4] https://datascience.stackexchange.com/questions/122571/determining-whether-a-dataset-is-imbalanced-or-not

### Regression

## Supervised Machine Learning Models and Their Training Algorithms
### Linear Regression Model
#### Gradient Descent
### Polynomial Regression Model
### Regularized Linear Models
### Logistic Regression Model
### Support Vector Machines
### Decision Trees
### Ensemble Learning and Random Forests