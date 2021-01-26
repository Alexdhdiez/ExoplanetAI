from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def plot_net(model):
    """
    Grafica y guarda la imagen con la estructura de la red

    Parameters
    ----------
    model : NN.MLP_res/NN.MLP/LSTM.LSTM_model/SOM2.SOM
        Modelo creado con las otras clases del proyecto
    """

    plot_model(model.get_model(), to_file = '../models/images/model_'+
                                            model.get_name()+'.png', show_shapes = True)

def plot_roc(model, name, labels, predictions, **kwargs):
    """
    Grafica y guarda la curva roc para unas etiquetas y predicciones

    Parameters
    ----------
    model : NN.MLP_res/NN.MLP/LSTM.LSTM_model/SOM2.SOM
        Modelo creado con las otras clases del proyecto
    name : str
        Nombre que se pondrá al pie de la imagen
    labels : array-like
        Etiquetas reales de los datos
    predictions : array-like
        Predicciones a los datos
    **kwargs
        Otros argumentos para el gráfico.

    """

    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,100.5])
    plt.ylim([-0.5,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.savefig('../models/images/ROC_'+
                model.get_name()+'.png')

def plot_history(model):
    """
    Grafica y guarda la imagen de las funciones de perdida del modelo

    Parameters
    ----------
    model : NN.MLP_res/NN.MLP/LSTM.LSTM_model/SOM2.SOM
        Modelo creado con las otras clases del proyecto
    """
    history = model.get_history()

    plt.figure()
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Val', linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../models/images/LOSS_'+
                model.get_name()+'.png')

def get_scores(labels, predictions):
    """
    Calcula las medidas de rendimiento de un modelo en base a las predicciones y a las
    clases reales

    Parameters
    ----------
    labels : array-like
        Etiquetas reales de los datos
    predictions : array-like
        Predicciones a los datos

    Returns
    -------
    dict of {str:int}
        Diccionario con los valores de precision, score y AUC
    """
    tn, fp, fn, tp  = confusion_matrix(labels, np.round(predictions)).ravel()
    succ = (tp+tn)/(tn+fp+fn+tp)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    score = succ*(0.5*sens+0.5*spec)
    fp, tp, _ = roc_curve(labels, predictions)
    AUC = auc(fp,tp)
    acc = accuracy_score(labels, np.round(predictions))
    results = {
        "acc": acc,
        "score": score,
        "auc": AUC
     }
    return results

def get_recall(labels, predictions):
    """
    Calcula la sensibilidad de un modelo en base a la clase y a las predicciones

    Parameters
    ----------
    labels : array-like
        Etiquetas reales de los datos
    predictions : array-like
        Predicciones a los datos

    Returns
    -------
    float
        Sensibilidad del modelo
    """
    tn, fp, fn, tp  = confusion_matrix(labels, np.round(predictions)).ravel()
    print(tp,fn)
    recall = tp/(tp+fn)
    return recall
