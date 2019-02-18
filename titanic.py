import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import scikitplot as skplt

import sklearn.preprocessing as preprocessing
import seaborn as sns

titanic_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Taxa de sobreviventes e mortos no desastre
labels = ['Mortos', 'Sobreviventes']
val_counts = titanic_data.Survived.value_counts()

sizes = [val_counts[0], val_counts[1]]
colors = ['#57e8fc', '#fc5e57']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, shadow=True, startangle=90, explode=(0.1,0), autopct='%1.1f%%', colors=colors)
ax.axis('equal')
plt.title('Porcentagem de sobreviventes e mortos no acidente')
plt.show()

# Tratamentos dados ao dataset:
# 1 - Retirada das colunas PassengerId, Cabin, Ticket e Name
# 2 - Transformar features Sex e Embarked em features numéricas, mantendo a sua categorização
# 3 - Criação de uma feature chamada isAlone, aonde a mesma é uma feature extraída das colunas SibSp and Parch
# 4 - Completar os valores faltantes em Age e Fare com a média

full_data = [titanic_data, test_data]

for dataset in full_data:
    # 1
    dataset.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], axis=1, inplace = True)
    # 2
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
    dataset['Embarked'].fillna(titanic_data['Embarked'].dropna().mode()[0], inplace = True)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # 3
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1, inplace=True)
    # 4
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

# retira a coluna de rótulos    
y_values = titanic_data['Survived'].values
titanic_data.drop(['Survived'], axis=1, inplace=True)

# converte DataFrame em Numpy
titanic_data=titanic_data.values


x_train, x_test, y_train, y_test = train_test_split(titanic_data, y_values, test_size=0.2, stratify=y_values, random_state=32)

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from time import process_time
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.metrics import roc_auc_score
import scikitplot as skpl
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

epocas = 100

# definir arquitetura MLP - com uma camada intermediária
mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=epocas, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01)

# definir arquitetura da SVM
svm = svm.SVC(kernel='rbf',C =1, gamma='auto')

# definir arquitetura da Random Forest
randF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=1)

# Aqui fazer um treinamento de x épocas e uma validacao da MLP
start = process_time()
mlp.fit(x_train, y_train)
end = process_time()
time_mlp = end - start
print('Métricas do treinamento da MLP')
print('Tempo de treinamento_mlp com ' + str (epocas) + ' épocas: ' + str(time_mlp))      
print("Erro no final do treinamento: %f" % mlp.loss_)
    
# Métricas da validacão MLP
preds_val_mlp = mlp.predict(x_test)  
print ('Métricas de uma validação MLP')
print("Acertos do conjunto de validação: %f" % mlp.score(x_test, y_test))
cm_val_mlp = confusion_matrix(y_test, preds_val_mlp)
print('Matriz de Confusão')
print(cm_val_mlp)
TP = cm_val_mlp[0,0]
FP = cm_val_mlp[0,1]
FN = cm_val_mlp[1,0]
TN = cm_val_mlp[1,1]

acuracia_mlp = (TP+TN)*100/(len(y_test))
precisao_mlp = TP*100/(TP+FP)
logloss_mlp = log_loss(y_test, preds_val_mlp)
roc_auc_mlp = roc_auc_score(y_test, preds_val_mlp)
print('acurácia_mlp:  '+ str(acuracia_mlp))
print('precisao_mlp:  '+ str(precisao_mlp))
print('logloss_mlp:  '+ str(logloss_mlp)) 
print('AUC_mlp:  '+ str(roc_auc_mlp*100)) 

preds_val_mlp_ = to_categorical(preds_val_mlp, num_classes=None)
skplt.metrics.plot_roc(y_test, preds_val_mlp_)
plt.show()

# Aqui fazer um treinamento e validacao da SVM  
start = process_time()
svm.fit(x_train, y_train)
end = process_time()
time_svm = end - start
print('Métricas do treinamento da SVM')
print('Tempo de treinamento_svm com ' + str (epocas) + ' épocas: ' + str(time_svm))

# Métricas da validacão SVM
preds_val_svm = svm.predict(x_test)  
print ('Métricas de uma validação SVM')
print("Acertos do conjunto de validação: %f" % svm.score(x_test, y_test))
cm_val_svm = confusion_matrix(y_test, preds_val_svm)
print('Matriz de Confusão')
print(cm_val_svm)
TP = cm_val_svm[0,0]
FP = cm_val_svm[0,1]
FN = cm_val_svm[1,0]
TN = cm_val_svm[1,1]

acuracia_svm = (TP+TN)*100/(len(y_test))
precisao_svm = TP*100/(TP+FP)
logloss_svm = log_loss(y_test, preds_val_svm)
roc_auc_svm = roc_auc_score(y_test, preds_val_svm)
print('acurácia_svm:  '+ str(acuracia_svm))
print('precisao_svm:  '+ str(precisao_svm))
print('logloss_svm:  '+ str(logloss_svm)) 
print('AUC_svm:  '+ str(roc_auc_svm*100)) 

preds_val_svm_ = to_categorical(preds_val_svm, num_classes=None)
skplt.metrics.plot_roc(y_test, preds_val_svm_)
plt.show() 

# Aqui fazer um treinamento e validacao da Random Forest  
start = process_time()
randF.fit(x_train, y_train)
end = process_time()
time_randF = end - start
print('Métricas do treinamento da Random Forest')
print('Tempo de treinamento_randF com ' + str (epocas) + ' épocas: ' + str(time_randF))

# Métricas da validacão randF
preds_val_randF= randF.predict(x_test)
print ('Métricas de uma validação Random Forest')
print("Acertos do conjunto de validação: %f" % randF.score(x_test, y_test))
cm_val_randF = confusion_matrix(y_test, preds_val_randF)
print('Matriz de Confusão')
print(cm_val_randF)
TP = cm_val_randF[0,0]
FP = cm_val_randF[0,1]
FN = cm_val_randF[1,0]
TN = cm_val_randF[1,1]
acuracia_randF = (TP+TN)*100/(len(y_test))
precisao_randF = TP*100/(TP+FP)
logloss_randF = log_loss(y_test, preds_val_randF)
roc_auc_randF = roc_auc_score(y_test, preds_val_randF)
print('acurácia_randF:  '+ str(acuracia_randF))
print('precisao_randF:  '+ str(precisao_randF))
print('logloss_randF:  '+ str(logloss_randF)) 
print('AUC_randF:  '+ str(roc_auc_randF*100)) 

preds_val_randF_ = to_categorical(preds_val_randF, num_classes=None)
skplt.metrics.plot_roc(y_test, preds_val_randF_)
plt.show() 
