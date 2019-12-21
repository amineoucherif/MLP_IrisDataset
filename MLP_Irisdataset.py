from sklearn.neural_network import MLPClassifier
from pandas import read_csv as rc
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from seaborn import kdeplot
from sklearn.metrics import confusion_matrix
from seaborn import set, heatmap

#Loading the data
dataset_init=rc('iris.csv',names=['Setal_length','Setal_width','Petal_length','Petal_width','Class'])
dataset=dataset_init.values
X=dataset[:,0:4] #Attributes
Y=dataset[:,4]   #True outputs
#Classify for the whole dataset
mlp=MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=800)
mlp.fit(X,Y)
pred=mlp.predict(X)
score=mlp.score(X,Y)
#Confusion matrix
cm=confusion_matrix(Y,pred, labels=["Iris-setosa","Iris-versicolor","Iris-virginica"])
df_cm = df(cm, range(3), range(3))
set(font_scale=1.4)#for label size
heatmap(df_cm, annot=True, annot_kws={"size": 20})
pyplot.xlabel("Predicted Output")
pyplot.ylabel("True Output")
pyplot.show()

#Spliting data
x_train, x_val, y_train, y_val=train_test_split(X,Y, test_size=0.3, random_state=0)
#Training set
mlp=MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=800)
mlp.fit(x_train,y_train) #returns a trained mlp model in form of object named 'mlp'
predictions_train=mlp.predict(x_train)
accuracy_train=mlp.score(x_train, y_train)
print "Training set accuracy : ",accuracy_train
#Validation set
predictions_val=mlp.predict(x_val)
accuracy_val=mlp.score(x_val, y_val)
print "Validation set accuracy : ",accuracy_val

#Data Visualization
#Histograms
dataset_init.hist()
pyplot.show()
#Scatter_matrix
scatter_matrix(dataset_init)
pyplot.show()
