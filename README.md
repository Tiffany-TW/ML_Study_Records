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
7. The Reciever Operating Characteristics (ROC) curve
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

*Features*:
1. $ 0 \leq \text{precision} \leq 1 $
2. The greater the value, the better the performance of the classifier

*Pros*:

*Cons*:

*Sci-Kit learn implementation*:
#### Recall for a specific class of the classifier
For a given class, recall of the classifier is an indicator for the sensitivity of the positive predictions, defined as $$\frac{TP}{TP+FN}$$ . In other words, recall is a measurement of how many positive instances are correctly predicted as positive, which is also called **sensitivity** or **TPR**.

*Features*:
1. $ 0 \leq \text{recall} \leq 1 $
2. The greater the value, the better the performance of the classifier

*Pros*:

*Cons*:

*Sci-Kit learn implementation*:
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


*Pros*:

*Cons*:

*Sci-Kit learn implementation*:

*Sci-Kit learn implementation*:

#### Precision/Recall (PR) Curve
Precision/Recall curve provides visualization for determining suitable thresholds for classifiers. Moreover, it shows apparent trends to illustrate the precision/recall trade-off.

*Features*:
1. The curve of a good classifier is closer to the top-right corner.
2. It is suitable for the cases where the positive class is rare or when we care more about the false positives than the false negatives. 

*Pros*:
*Cons*:
*Sci-Kit learn implementation*:

#### The Reciever Operating Characteristics (ROC) curve
The ROC curve plots **recall** against **false positive rate (FPR)**. FPR is defined as $\frac{FP}{TP+FP}$ . Actually, FPR is also equal to 1 - TNR (true negative rate, called specificity). TNR is defined as $\frac{TN}{TN+FN}$ .

*Features*:
1. There is trade-off between recall (TPR) and FPR. That is, the higher the recall (TPR), the more FPR the classifier produces.
2. The ROC curve of a good classifier is closer to the top-left corner.
3. The area under the curve (AUC) of a ROC curve of a purely random classifier is 0.5. For a perfect classifier, its AUC is equal to 1.

*Pros*:
*Cons*:
*Sci-Kit learn implementation*:
#### Supplementary
[1] Having similar precision and recall is not always the case. Sometimes, we prefer precision over recall. Sometimes, we care more about recall. It depends on the contexts. Please read p.111 for examples.

[2] Because of the precision/recall trade-off, it is impossible for us to have high values both ways: Please read [Ref](#reference) [3] for more theoretical explaination.

#### Reference
[1] https://www.evidentlyai.com/classification-metrics/multi-class-metrics

[2] Scikit-learn, L. W. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow. Ò Reilly Media.

[3] https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-4571(199401)45:1%3C12::AID-ASI2%3E3.0.CO;2-L

