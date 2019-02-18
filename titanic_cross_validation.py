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
k = 10

# definir arquitetura MLP - com uma camada intermediária
mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=epocas, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01)

# definir arquitetura da SVM
svm = svm.SVC(kernel='rbf',C =1, gamma='auto')

# definir arquitetura da Random Forest
randF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=1)

time_train_mlp =[]
acuracia_mlp = []
precisao_mlp = []
logloss_mlp = []
roc_auc_mlp = []

time_train_svm =[]
acuracia_svm = []
precisao_svm = []
logloss_svm = []
roc_auc_svm = []

time_train_randF =[]
acuracia_randF = []
precisao_randF = []
logloss_randF = []
roc_auc_randF = []

# Validação cruzada com k folds
skf = StratifiedKFold(n_splits=k, random_state=None)

for train_index, test_index in skf.split(titanic_data, y_values): 
    
    sinais_treinamento, sinais_validacao = titanic_data[train_index], titanic_data[test_index] 
    labels_treinamento, labels_validacao = y_values[train_index], y_values[test_index]
    
    x_treinamento, y_treinamento = shuffle(sinais_treinamento, labels_treinamento, random_state = 42)
    x_validacao, y_validacao = shuffle(sinais_validacao, labels_validacao, random_state = 42)
    
# Aqui fazer um treinamento de x épocas e uma validacao da MLP
    start = process_time()
    mlp.fit(x_treinamento, y_treinamento)
    end = process_time()
    time_mlp = end - start
    print('Métricas do treinamento da MLP')
    print('Tempo de treinamento_mlp com ' + str (epocas) + ' épocas: ' + str(time_mlp))      
    print("Erro no final do treinamento: %f" % mlp.loss_)
      
# Métricas da validacão mlp
    preds_val_mlp = mlp.predict(x_validacao)  
    print ('Métricas de uma validação MLP')
    print("Acertos do conjunto de validação: %f" % mlp.score(x_validacao, y_validacao))
    cm_val_mlp = confusion_matrix(y_validacao, preds_val_mlp)
    print('Matriz de Confusão')
    print(cm_val_mlp)
    TP = cm_val_mlp[0,0]
    FP = cm_val_mlp[0,1]
    FN = cm_val_mlp[1,0]
    TN = cm_val_mlp[1,1]

    acuracia_mlp_ = (TP+TN)*100/(len(y_validacao))
    precisao_mlp_ = TP*100/(TP+FP)
    logloss_mlp_ = log_loss(y_validacao, preds_val_mlp)
    roc_auc_mlp_ = roc_auc_score(y_validacao, preds_val_mlp)
    print('acurácia_mlp_:  '+ str(acuracia_mlp_))
    print('precisao_mlp_:  '+ str(precisao_mlp_))
    print('logloss_mlp_:  '+ str(logloss_mlp_)) 
    print('AUC_mlp_:  '+ str(roc_auc_mlp_*100)) 

# Usar no calculo das médias da mlp
    time_train_mlp.append(time_mlp)
    acuracia_mlp.append(acuracia_mlp_)
    precisao_mlp.append(precisao_mlp_)
    logloss_mlp.append(logloss_mlp_)
    roc_auc_mlp.append(roc_auc_mlp_*100)
       
# Aqui fazer um treinamento e validacao da SVM  
    start = process_time()
    svm.fit(x_treinamento, y_treinamento)
    end = process_time()
    time_svm = end - start
    print('Métricas do treinamento da SVM')
    print('Tempo de treinamento_svm com ' + str (epocas) + ' épocas: ' + str(time_svm))
# Métricas da validacão SVM
    preds_val_svm= svm.predict(x_validacao)
    print ('Métricas de uma validação SVM') 
    print("Acertos do conjunto de validação: %f" % svm.score(x_validacao, y_validacao))
    cm_val_svm = confusion_matrix(y_validacao, preds_val_svm)
    print('Matriz de Confusão')
    print(cm_val_svm)
    TP = cm_val_svm[0,0]
    FP = cm_val_svm[0,1]
    FN = cm_val_svm[1,0]
    TN = cm_val_svm[1,1]
    
    acuracia_svm_ = (TP+TN)*100/(len(y_validacao))
    precisao_svm_ = TP*100/(TP+FP)
    logloss_svm_ = log_loss(y_validacao, preds_val_svm)
    roc_auc_svm_ = roc_auc_score(y_validacao, preds_val_svm)
    print('acurácia_svm_:  '+ str(acuracia_svm_))
    print('precisao_svm_:  '+ str(precisao_svm_))
    print('logloss_svm_:  '+ str(logloss_svm_)) 
    print('AUC_svm_:  '+ str(roc_auc_svm_*100)) 

# Usar no calculo das médias da svm
    time_train_svm.append(time_svm)
    acuracia_svm.append(acuracia_svm_)
    precisao_svm.append(precisao_svm_)
    logloss_svm.append(logloss_svm_)
    roc_auc_svm.append(roc_auc_svm_*100)
    
media_time_train_mlp = sum(time_train_mlp) / float(len(time_train_mlp))
media_acuracia_mlp = sum(acuracia_mlp) / float(len(acuracia_mlp))
media_precisao_mlp = sum(precisao_mlp) / float(len(precisao_mlp))
media_logloss_mlp = sum(logloss_mlp) / float(len(logloss_mlp))
media_roc_auc_mlp = sum(roc_auc_mlp) / float(len(roc_auc_mlp))

media_time_train_svm = sum(time_train_svm) / float(len(time_train_svm))
media_acuracia_svm = sum(acuracia_svm) / float(len(acuracia_svm))
media_precisao_svm = sum(precisao_svm) / float(len(precisao_svm))
media_logloss_svm = sum(logloss_svm) / float(len(logloss_svm))
media_roc_auc_svm = sum(roc_auc_svm) / float(len(roc_auc_svm))



print('Tempo médio de treinamento MLP com ' + str(k) + ' kfold ' + str (media_time_train_mlp))
print('Médias das Validações com ' + str(k) + ' folds')
print('Acurácia_mlp: ' + str(media_acuracia_mlp))
print('Precisão_mlp: ' + str(media_precisao_mlp))
print('LogLoss_mlp: ' + str(media_logloss_mlp))
print('AUC_mlp:  '+ str(media_roc_auc_mlp)) 

print('Tempo médio de treinamento SVM com ' + str(k) + ' kfold ' + str (media_time_train_svm))
print('Médias das Validações com ' + str(k) + ' folds')
print('Acurácia_svm: ' + str(media_acuracia_svm))
print('Precisão_svm: ' + str(media_precisao_svm))
print('LogLoss_svm: ' + str(media_logloss_svm))
print('AUC_svm:  '+ str(media_roc_auc_svm)) 
