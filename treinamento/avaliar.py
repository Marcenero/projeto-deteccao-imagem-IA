"""Metricas binarias, com FAKE como classe positiva."""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def metricas(y, previsto):
    matriz=confusion_matrix(y,previsto,labels=[0,1]); tn,fp,fn,tp=matriz.ravel()
    return {'acuracia':float(accuracy_score(y,previsto)), 'precisao':float(precision_score(y,previsto,zero_division=0)), 'recall':float(recall_score(y,previsto,zero_division=0)), 'f1':float(f1_score(y,previsto,zero_division=0)), 'especificidade':float(tn/(tn+fp)) if tn+fp else 0.0, 'matriz_confusao':matriz.tolist()}
