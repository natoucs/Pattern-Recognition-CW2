
#class: 1:59 2:71 3:48  & feature: 13
#1st col : 1/2 1-train 2-test - 2nd col: 1/2/3 classes

#Import data
import numpy as np
wine = np.genfromtxt('wine.data.csv', delimiter='', dtype=np.float)

#Store data
split = wine[:,0]
label = wine[:,1]
data = wine[:,2:]
X_train = data[split == 1]
X_test = data[split == 2]
y_train = label[split == 1]
y_test = label[split == 2]

#Data preprocessing
#Scale features of input vector X to [0,1] or make it mean 0 var 1
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#Train
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
                    activation="relu",
                    solver='sgd',
                    alpha=1e-5, #2?
                    #hidden_layer_sizes= (6),
                    hidden_layer_sizes= (13, 13, 13), 
                    learning_rate='adaptive', 
                    max_iter=200, 
                    random_state=1,
                    verbose=False,
                    warm_start=True
                    )

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#for i in range (30):
mlp.fit(X_train, y_train)
trainpred = mlp.predict(X_train)
predictions = mlp.predict(X_test)
print ("train score " , accuracy_score(y_train, trainpred))
print ("test score " , accuracy_score(y_test, predictions))

#Extract weights and prune empty neurons
#Monitor weight changing at each training iteration and monitor learning

#Test
#predictions = mlp.predict(X_test)

#Report results
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#print accuracy_score(y_train, trainpred)
#print accuracy_score(y_test, predictions)

print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))

#Extract MLP weights and biases after training
#print (mlp.coefs_)
#print len(mlp.coefs_[0])
#print len(mlp.intercepts_[0])