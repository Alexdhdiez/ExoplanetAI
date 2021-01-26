import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from dataRead import ReadData
import plots
import numpy as np
import pickle


class LSTM_model:
    """
    Clase que implementa el modelo de red neuronal LSTM y sus métodos relacionados

    Attributes
    ----------
    data : dataRead.ReadData
        Instancia de ReadData con los datos
    epochs : int
        Número de epocas para entrenar la red
    batch : int
        Tamaño de batch
    metrics : list of str
        Metricas a imprimir por pantalla
    loss : str
        Funcion de coste o pérdida. Acepta cualquiera de las in-built de Keras
    optim : str
        Optimizador del entrenamiento. Acepta cualquiera de los in-built de Keras.
    units : str
        Numero de celdas LSTM
    features : int
        Numero de características de los datos
    model : tensorflow.keras.models.Model
        Modelo de red creado o importado
    history : DataFrame
        Historia de modelos previamente creados y ajustados
    method : str
        metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced

    Methods
    -------
    create_model(final_activation = 'sigmoid'):
        Crea un modelo de red neuronal
    compile():
        Compila el modelo
    train():
        Entrena el modelo creado
    predict(test = False):
        Obtiene predicciones del conjunto especificado
    save_model():
        Guarda el modelo
    save_history():
        Guarda la historia del modelo
    """

    def __init__(self, units, path_train, path_test, features=1, optim = 'adam',
                    loss = 'binary_crossentropy', metrics = ['accuracy'],
                    epochs = 50, batch = 1, model = None, history = None, method = None):

        """
        Parameters
        ----------
        units : int
            Número de celdas LSTM
        path_train : str
            Ruta al archivo de datos de entrenamiento
        path_test : str
            Ruta al archivo de datos de prueba
        metrics : list of str
            Metricas a imprimir por pantalla
        loss : str
            Funcion de coste o pérdida. Acepta cualquiera de las in-built de Keras.
        optim : str
            Optimizador del entrenamiento. Acepta cualquiera de los in-built de Keras.
        units : str
            Numero de celdas LSTM
        features : int
            Numero de características de los datos
        model : tensorflow.keras.models.Model
            Modelo de red creado o importado de uno existente
        hist : DataFrame
            Historia de modelos previamente creados y ajustados
        method : str
            metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced
        epochs : int
            Número de epocas para entrenar la red
        batch : int
            Tamaño de batch
        """
        #Hiperparámetros
        self.epochs = epochs
        self.batch = batch
        self.metrics = metrics
        self.loss = loss
        self.optim = optim
        self.units = units
        self.features = features

        #Leemos los datos
        self.data = ReadData(path_train, path_test)
        self.data.adapt_features(features, self.data.X_train.shape[1])
        #Adaptamos al numero de features que queramos
        self.n_t = int(self.data.X_train.shape[1]/self.features)
        self.data.split(0.33)

        self.method = method
        if method != None:
            self.data.imbalanced(method=method)

        #Convertir datos en aptos para los modelos de Keras (tensores)
        self.data.X_val = self.data.X_val.reshape(self.data.X_val.shape[0], self.n_t, self.features)
        self.data.X_test = self.data.X_test.reshape(self.data.X_test.shape[0], self.n_t, self.features)
        self.data.X_train = self.data.X_train.reshape(self.data.X_train.shape[0], self.n_t, self.features)
        self.data.X_t = self.data.X_t.reshape(self.data.X_t.shape[0], self.n_t, self.features)

        #Creamos modelo en caso de que ya hubiera uno y se lo estemos pasando
        self.model = model
        self.hist = history

    def create_model(self, final_activation = 'sigmoid'):
        """
        Crea un modelo de red LSTM en funcion del numero de celdas

        Parameters
        ----------
        final_activation : str
            Activacion de la capa de salida de la red. Acepta cualquiera de las in-built de Keras.
        """

        # Entradas a la red, se indica el numero de timesteps y de caracteristicas
        inputs = Input(shape=(self.n_t, self.features))
        # return_sequences = False para red muchos a uno
        x = LSTM(units = self.units, return_sequences=False)(inputs)
        outputs = Dense(1, activation = final_activation)(x)

        self.model = Model(inputs = inputs, outputs = outputs, name = 'LSTM_'+str(self.units)+
                                                                    '_'+str(self.batch)+
                                                                    '_'+str(self.epochs)+
                                                                    '_'+str(self.method))

    def compile(self):
        """
        Compila el modelo
        """
        self.model.compile(loss = self.loss, optimizer=self.optim,
                            metrics=self.metrics)

    def train(self):
        """
        Entrena el modelo utilizando como datos de validacion los obtenidos en la division
        de los conjuntos en los datos
        """
        history = self.model.fit(self.data.X_t, self.data.y_t,
                    validation_data = (self.data.X_val, self.data.y_val),
                    epochs = self.epochs, batch_size = self.batch,
                    class_weight = self.data.class_weight, shuffle=False)
        self.hist = history.history

    def predict(self, test = False):
        """
        Calcula las predicciones para el conjunto especificado

        Parameters
        ----------
        test : bool
            Valor booleano indicativo de si se quiere predecir el conjunto de prueba (True)
            o el de entrenamiento (False)

        Returns
        -------
        array-like
            array con las predicciones para el conjunto que fuera especificado
        """
        if test:
            prediction = self.model.predict(self.data.X_test)
        else: #Queremos predecir todo el conjunto de entrenamiento no solo el 66% de HO
            prediction = self.model.predict(self.data.X_train)

        return prediction

    def save_model(self):
        """
        Guarda el modelo
        """
        tf.keras.models.save_model(self.model,
                                    '../models/LSTM/'+self.model.name+
                                    '.h5')
    def save_history(self):
        """
        Guarda la historia del modelo
        """
        with open('../models/LSTM/'+self.model.name, 'wb') as file_pi:
            pickle.dump(self.hist, file_pi)

    def get_history(self):
        return self.hist

    def get_labels_test(self):
        return self.data.y_test

    def get_labels_train(self):
        return self.data.y_train

    def get_n(self):
        return self.units

    def get_optim(self):
        return self.optim

    def get_epochs(self):
        return self.epochs

    def get_loss(self):
        return self.loss

    def get_f(self):
        return self.features

    def get_model(self):
        return self.model

    def get_name(self):
        return self.model.name
