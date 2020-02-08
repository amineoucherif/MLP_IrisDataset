# MLP_IrisDataset
This repository contains a script code which classifies the famous Iris Dataset.
Here we are generating a machine learning algorithm based on the MLP artificial neural network architecture, to classify the 3 types of the Iris species.
This dataset contains 150 samples, each sample is caracterized by 4 attributes.
+ Python 2.7.17
+ Required libraries : sklearn, pandas, matplotlib, seaborn

## Quick Data Visualization
### Histograms
![Histograms](https://github.com/amineoucherif/MLP_IrisDataset/blob/master/Histograms.png)

### Scatter Matrix
![Scatter Matrix](https://github.com/amineoucherif/MLP_IrisDataset/blob/master/ScatterMatrix.png)

## Results
Note that after each execution the results may variate, below represented the results obtained in one specific execution.

### Accuracy
![Accuracy](https://github.com/amineoucherif/MLP_IrisDataset/blob/master/Accuracy.png)

### Confusion Matrix
In supervised learning, we use the confusion matrix to mesure the quality of the classifier.

As we said before, there are 3 classes.

Knowing that each class is represented by an integer : 0=Iris-setosa, 1=Iris-versicolor, 2=Iris-virginica.
Each  one of these classes, contains 50 samples.
Below is the confusion matrix resulted from this execution :
![Confusion Matrix](https://github.com/amineoucherif/MLP_IrisDataset/blob/master/ConfusionMatrix.png)
Let's take for instance the first class, Iris-setosa, which is represented by 0.
Our algorithm didn't never predict the output as 0, whereas the true output is 1 or 2.
On the other hand it predicted 50 times the output as 0 whereas it's trully 0. In other words for this class 0 our algorithm were 100% accurate.

## Conclusion
We observe that using MLP architecture on Iris dataset permited the obtention of persuasive results.

###### Bibliography
Iris dataset : https://archive.ics.uci.edu/ml/datasets/iris


