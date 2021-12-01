import pandas as pd, numpy as np
tf.keras.utils.set_random_seed(42)

import json
import csv
import random
import numpy as np
import sklearn
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random

# Separate dataset in train and test

def separate_train_and_test(df, class_column ,sub_classes_toTakeOff=[], sub_classes_toKeep=[], seed=42, percent_sample=0.1, sample_index=1):
  train_samples = []
  test_samples = [] # A test_sample it's gonna be all the dataset without a the elementes from train
  
  if sample_index*percent_sample > 1:
    print("ERRO: Invalide sample Index")
    return [], []

  df_without_subclasses = df
  
  # Cut of the subclasses we don't need
  for subclass in sub_classes_toTakeOff:
    df_without_subclasses = df[df[subclass] != 1]

  for subclass in sub_classes_toKeep:
    df_without_subclasses = df[df[subclass] == 1]

  
  df_without_subclasses = shuffle(df_without_subclasses, random_state=seed)
  tam_new_df = df_without_subclasses.shape[0]

  #Getting the samples, doing manual stratification
  df2 = df_without_subclasses[df_without_subclasses[class_column] == 1]
  tam_df2 = df2.shape[0]
  df_train2 = df2[int(percent_sample*tam_df2*(sample_index-1)):int(percent_sample*tam_df2*(sample_index))]
  df_test2  = df.loc[df[class_column] == 1].drop(df_train2.index)

  df3 = df_without_subclasses[df_without_subclasses[class_column] == 0]
  tam_df3 = df3.shape[0]
  df_train3 = df3[int(percent_sample*tam_df3*(sample_index-1)):int(percent_sample*tam_df3*(sample_index))]
  df_test3  = df.loc[df[class_column] == 0].drop(df_train3.index)

  #Juntar
  df_train = pd.concat([df_train3, df_train2])
  df_test = pd.concat([df_test3, df_test2])

  #aleatorizar
  df_train = shuffle(df_train, random_state=seed)
  df_test = shuffle(df_test, random_state=seed)

  return df_train, df_test



def separate_train_validation(df_train, class_column, percent=0.7, seed=12):
   from sklearn.model_selection import train_test_split
   X_t, X_val, y_t, y_val = train_test_split(df_train, df_train[class_column], train_size=percent, random_state=seed)

   return X_t, X_val

def class_size_graph(Y_train,Y_test, Y_val):
  labels = ["%s"%i for i in range(3)]

  unique, counts = np.unique(Y_train, return_counts=True)
  uniquet, countst = np.unique(Y_test, return_counts=True)
  uniquev, countsv = np.unique(Y_val, return_counts=True)

  fig, ax = plt.subplots()
  rects3 = ax.bar(uniquev - 0.5, countsv, 0.25, label='Validation')
  rects1 = ax.bar(unique - 0.2, counts, 0.25, label='Train')
  rects2 = ax.bar(unique + 0.1, countst, 0.25, label='Test')
  ax.legend()
  ax.set_xticks(unique)
  ax.set_xticklabels(labels)

  plt.title('Hate Speech classes')
  plt.xlabel('Class')
  plt.ylabel('Frequency')
  plt.show()

def plot_confusion_matrix(y, y_pred, beta = 2):
    """
    It receives an array with the ground-truth (y)
    and another with the prediction (y_pred), both with binary labels
    (positve=+1 and negative=-1) and plots the confusion
    matrix.
    It uses P (positive class id) and N (negative class id)
    which are "global" variables ...
    """
    TP = np.sum((y_pred == 1) * (y == 1))
    TN = np.sum((y_pred == 0) * (y == 0))

    FP = np.sum((y_pred == 1) * (y == 0))
    FN = np.sum((y_pred == 0) * (y == 1))

    total = TP+FP+TN+FN

    accuracy = (TP+TN)/total
    recall = (TP)/(TP+FN)
    precision = (TP)/(TP+FP)

    Fbeta = (precision*recall)*(1+beta**2)/(beta**2*precision + recall)
    
    print("TP = %4d    FP = %4d\nFN = %4d    TN = %4d\n"%(TP,FP,FN,TN))
    print("Accuracy = %d / %d (%f)" %((TP+TN),total, (TP+TN)/total))
    print("Recall = %d / %d (%f)" %((TP),(TP+FN), (TP)/(TP+FN)))
    print("Precision = %d / %d (%f)" %((TP),(TP+FP), (TP)/(TP+FP)))
    print("Fbeta Score = %f" %(Fbeta))

    confusion = [
        [TP/(TP+FN), FP/(TN+FP)],
        [FN/(TP+FN), TN/(TN+FP)]
    ]

    P = 1
    N = 0

    df_cm = pd.DataFrame(confusion, \
                         ['$\hat{y} = %d$'%P, '$\hat{y} = %d$'%N],\
                         ['$y = %d$'%P, '$y = %d$'%N])
    
    plt.figure(figsize = (8,4))
    sb.set(font_scale=1.4)
    sb.heatmap(df_cm, annot=True) #, annot_kws={"size": 16}, cmap = 'coolwarm')
    plt.show()