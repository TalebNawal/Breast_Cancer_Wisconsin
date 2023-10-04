#importation de librairies
from datetime import time
from joblib import dump, load
import time
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

"""loading the data"""
btwData = pd.read_csv("C:/Users/pc/Desktop/Projet Integré S4/data.csv")


"""data info"""
#information sur la data
print("informations sur les données",btwData.info())

# Affichage des premiers lignes de btwData
print(btwData.head())

# Information sur la data pour voir les valeurs à standariser
print("Afficher les statistiques des données",btwData.describe().T)

#nombre de valeur unique en la colonne diagnosis
print(btwData.diagnosis.unique())

# colonne diagnosis
print(btwData['diagnosis'].value_counts())
plt.hist(btwData['diagnosis'])
plt.show()


"""#################Data cleaning and preparing####################"""

#droping the id and empty columns
btwData.drop('id',axis=1,inplace=True)
btwData.drop('Unnamed: 32',axis=1,inplace=True)

print(btwData.head())

#changing M by 1 and B by 0
btwData['diagnosis']= btwData['diagnosis'].replace(['M','B'],[1,0])
print(btwData.head())

#Voir le nombre des valeurs nuls
print(btwData.isnull().sum())

#correlation entre les variables
print(btwData.corr())
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(btwData.corr(), annot=True, ax=ax)
plt.show()

#separation des means /se/worst
columns=['diagnosis', 'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean'
    ,'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean' ]
sns.pairplot(data=btwData[columns],hue='diagnosis',palette='rocket')
plt.show()
"""probleme de multicoliniarité entre radius, parameter, area"""
"""solution de laisser qu'une seule variable pour bien distinguer l'effet sue la determination de cancer"""

"""aussi concavity, concavity points"""


hide = np.zeros_like(btwData.corr().round(2))
hide[np.triu_indices_from(hide)]=True
fig, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(btwData.corr(),mask=hide,cmap=sns.diverging_palette(220,10,as_cmap=True),vmin=-1,vmax=1,center=0,
            square=True,linewidths=5,cbar_kws={"shrink":5},annot=True)
plt.show()

columns=['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst'
    ,'concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst' ]

btwData = btwData.drop(columns,axis=1)

columns = ['perimeter_mean','area_mean','concavity_mean','concave points_mean','perimeter_se','area_se','concavity_se'
    ,'concave points_se']
btwData = btwData.drop(columns,axis=1)
print(btwData.columns)


hide = np.zeros_like(btwData.corr().round(2))
hide[np.triu_indices_from(hide)]=True
fig, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(btwData.corr(),mask=hide,cmap=sns.diverging_palette(220,10,as_cmap=True),vmin=-1,vmax=1,center=0,
            square=True,linewidths=5,cbar_kws={"shrink":5},annot=True)
plt.show()

### build the model
X= btwData.drop(['diagnosis'],axis=1)

y=btwData['diagnosis']

#feature scaling
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
print("data cloumns")
print(X.columns)
standardScaler=StandardScaler()
X_train=standardScaler.fit_transform(X_train)
X_test=standardScaler.fit_transform(X_test)



"""Logistic Regression"""
lr = LogisticRegression()
start = time.time()
model1 = lr.fit(X_train,y_train)
end=time.time()
start_p = time.time()
prediction1 = model1.predict(X_test)

end_p = time.time()
cm = confusion_matrix(y_test,prediction1)
print("***************logistic model**********************")
#matrice de confusion
print(cm)
sns.heatmap(cm,annot=True)

FP = cm[0][1]
FN = cm[0][0]
TP = cm[1][1]
TN = cm[1][0]
# calcul de precision
precision=TP/(TP+FP)
#calcul d'accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
#calcul de recall
recall= TP/(TP+FN)
#calcul de f1-score
f1Score= 2*((recall*precision)/(recall+precision))
print("F1 Score for logistic regression model is:",f1Score)
print("accuracy for logistic regression model is:",accuracy)
print("precision for logistic regression model is:",precision)
print("recall for logistic regression model is:",recall)
print("Training time for logistic regression model is:", end - start, "seconds")
print("Prediction time for logistic regression model is:", end_p - start_p, "seconds")

"""
K Nearest Neighbor (K NN)
Support Vector Machine
Naive Bayes
"""
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
for name, model in models:
    # Train the model on the training data
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    # Make predictions on the test data
    start_pred_time = time.time()
    y_pred = model.predict(X_test)
    end_pred_time = time.time()

    # Compute the performance metrics of the model on the test data
    accuracy = accuracy_score(y_test, y_pred)
    f1_scor = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Print the performance metrics and time of the model on the test data
    print("***************", name, "model**********************")
    print("Accuracy value for", name, "is:", accuracy)
    print("F1-score value for", name, "is:", f1_scor)
    print("Recall value for", name, "is:", recall)
    print("Precision value for", name, "is:", precision)
    print("Training time for", name, "is:", end_time - start_time, "seconds")
    print("Prediction time for", name, "is:", end_pred_time - start_pred_time, "seconds")

    # Append the accuracy of the model to the results list
    results.append(accuracy)

    # Plot the confusion matrix as a heatmap
    cm = confusion_matrix(y_test, y_pred)
    cm[np.isnan(cm)] = 0
    cm[np.isinf(cm)] = 0
    sns.heatmap(cm, annot=True)
    plt.savefig(name)
    plt.show()
print(models)
dump(models[0][1], 'knn_model.joblib')
"""Decision tree"""

dtc=DecisionTreeClassifier()
start = time.time()
model3=dtc.fit(X_train,y_train)
end = time.time()
start_p=time.time()
prediction3=model3.predict(X_test)
print(X_test)
end_p=time.time()
cm3= confusion_matrix(y_test,prediction3)

print("***************Decision tree model**********************")
print(cm3)
print("Accuracy value for Decision tree model is: ",accuracy_score(y_test,prediction3))
print("recall value for Decision tree model is: ",recall_score(y_test,prediction3))
print("precision value for Decision tree model is: ",precision_score(y_test,prediction3))
print("F1-score value for Decision tree model is: ",f1_score(y_test,prediction3))
print("Training time for Decision tree model is:", end - start, "seconds")
print("Prediction time for Decision tree model is:", end_p - start_p, "seconds")




