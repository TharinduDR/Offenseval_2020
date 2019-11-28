from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


def evaluatation_scores(test, target_label, prediction_label):
    confusion_matrix_values = confusion_matrix(test[prediction_label], test[target_label]).ravel()
    accuracy = accuracy_score(test[prediction_label], test[target_label])
    weighted_f1 = f1_score(test[target_label], test[prediction_label], average='weighted')
    weighted_recall = recall_score(test[target_label], test[prediction_label], average='weighted')
    weighted_precision = precision_score(test[target_label], test[prediction_label], average='weighted')

    return confusion_matrix_values, accuracy, weighted_f1, weighted_recall, weighted_precision