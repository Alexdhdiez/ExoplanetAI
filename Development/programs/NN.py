import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LeakyReLU, ReLU
from tensorflow.keras.models import Model, Sequential
from dataRead import ReadData
import plots
import numpy as np
import pickle

class MLP_res:
    """
    Clase que implementa el modelo de red neuronal plenamente conectada con capas residuales

    Attributes
    ----------
    data : dataRead.ReadData
        Instancia de ReadData con los datos
    epochs : int
        Número de epocas para entrenar la red
    batch : int
        Tamaño de batch
    metrics : list of str
        Metricas a imprimir por pantalla. Acepta cualquiera de las in-built de Keras.
    loss : str
        Funcion de coste o pérdida. Acepta cualquiera de las in-built de Keras.
    optim : str
        Optimizador del entrenamiento. Acepta cualquiera de los in-built de Keras.
    n_layer : int
        Número de capas de la red
    n_neuron : list of int
        Lista con las neuronas de cada capa
    alpha : float
        alpha utilizado para la activacion LeakyReLU
    res_size : int
        Tamaño de los residuales
    model : tensorflow.keras.models.Model
        Modelo de red creado o importado
    history : DataFrame
        Historia de modelos previamente creados y ajustados
    method : str
        Metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced

    Methods
    -------
    create_model(hidden_activation="sigmoid", final_activation = 'sigmoid'):
        Crea un modelo de red neuronal con las activaciones ocultas y finales indicadas
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

    def __init__(self, n_layer, n_neuron, path_train, res_size,
                path_test, optim = 'adam', loss = 'binary_crossentropy',
                metrics = ['binary_accuracy'], epochs = 50, batch = 1,
                model = None, history=None, alpha=0.01, method=None):
        """
        Parameters
        ----------
        n_layer : int
            Número de capas de la red
        n_neuron : list of int
            Lista con las neuronas de cada capa
        alpha : float
            alpha utilizado para la activacion LeakyReLU
        res_size : int
            Tamaño de los residuales
        path_train : str
            Ruta al archivo de datos de entrenamiento
        path_test : str
            Ruta al archivo de datos de prueba
        metrics : list of str
            Metricas a imprimir por pantalla. Acepta cualquiera de las in-built de Keras.
        loss : str
            Funcion de coste o pérdida. Acepta cualquiera de las in-built de Keras.
        optim : str
            Optimizador del entrenamiento. Acepta cualquiera de los in-built de Keras.
        model : tensorflow.keras.models.Model
            Modelo de red creado o importado de uno existente
        hist : DataFrame
            Historia de modelos previamente creados y ajustados
        method : str
            Metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced
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
        self.n_layer = n_layer
        self.n_neuron = n_neuron
        self.alpha = alpha          #alpha for LeakyRelu
        self.res_size = res_size    #Size of the residuals

        #Leemos los datos
        self.data = ReadData(path_train, path_test)
        self.data.split(0.33)

        self.method = method
        if method != None:
            self.data.imbalanced(method=method)

        #Creamos modelo en caso de que ya hubiera uno y se lo estemos pasando
        self.model = model
        self.hist = history

    def create_model(self, hidden_activation="sigmoid", final_activation = 'sigmoid'):
        """
        Crea un modelo de red con residuales basandose en los parámetros indicados

        Parameters
        ----------
        hidden_activation : str
            Activacion de las capas ocultas de la red. Acepta cualquiera de las in-built de Keras.
        final_activation : str
            Activacion de la capa de salida de la red. Acepta cualquiera de las in-built de Keras.
        """

        # Capa de entrada. Se define la forma con el tamaño de cada observacion
        inputs = Input(shape=self.data.X_t.shape[1])

        first = True
        x = inputs

        # Tantas capas como se indiquen
        for l in range(0,self.n_layer):
            n = self.n_neuron[l]

            # Se añade la activacion indicada en los parámetros
            if hidden_activation=="lrelu":
                x = Dense(n)(x)                    #Capa plenamente conectada
                x = LeakyReLU(alpha=self.alpha)(x) #Le damos la activacion
            elif hidden_activation=="relu":
                x = Dense(n, activation="relu")(x)
            elif hidden_activation=="sigmoid":
                x = Dense(n, activation="sigmoid")(x)
            else:
                x = Dense(n, activation="tanh")(x)

            # Solo cada capa tras pasar el tamaño de residual se guarda el bloque
            if (l%self.res_size == 0):
                #Si es la primera vez el bloque solo guardará la capa
                if first:
                    block = x
                    first = False
                else:
                    #Guardamos el estado del residual
                    block = tf.keras.layers.add([x, block])
                x = block

        # Capa de salida
        outputs = Dense(1, activation=final_activation)(x)
        self.model = Model(inputs = inputs, outputs = outputs, name = "MLP_res"
                                                                    + hidden_activation+'_'
                                                                    + final_activation+'_'
                                                                    + str(self.n_layer)+'_'+'-'
                                                                    +str(self.n_neuron).strip('[], ').replace(', ','-')+'-'+'_'
                                                                    +str(self.res_size)+'_'
                                                                    +str(self.epochs)+'_'
                                                                    +str(self.method))

    def compile(self):
        """
        Compila el modelo
        """
        self.model.compile(loss=self.loss, optimizer=self.optim)

    def train(self):
        """
        Entrena el modelo utilizando como datos de validacion los obtenidos en la division
        de los conjuntos en los datos
        """
        history = self.model.fit(self.data.X_t, self.data.y_t,
                    validation_data = (self.data.X_val, self.data.y_val),
                    epochs = self.epochs, batch_size = self.batch,
                    class_weight = self.data.class_weight)
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
            Array con las predicciones para el conjunto que fuera especificado
        """
        if test:
            prediction = self.model.predict(self.data.X_test)
        else: #Queremos predecir todo el conjunto de entrenamiento no solo el 66% de HO
            prediction = self.model.predict(self.data.X_train)

        return prediction

    def save_model(self):
        tf.keras.models.save_model(self.model,
                                    '../models/Perceptron/'+
                                    self.model.name+'.h5')
    def save_history(self):
        with open('../models/Perceptron/History'+self.model.name, 'wb') as file_pi:
            pickle.dump(self.hist, file_pi)

    def get_labels_test(self):
        return self.data.y_test

    def get_labels_train(self):
        return self.data.y_train

    def get_n(self):
        return self.n_layer

    def get_optim(self):
        return self.optim

    def get_epochs(self):
        return self.epochs

    def get_loss(self):
        return self.loss

    def get_history(self):
        return self.hist

    def get_f(self):
        return self.res_size

    def get_model(self):
        return self.model

    def get_name(self):
        return self.model.name


class MLP:
    """
    Clase que implementa el modelo de red neuronal plenamente conectada o perceptron multicapa

    Attributes
    ----------
    data : dataRead.ReadData
        Instancia de ReadData con los datos
    epochs : int
        Número de epocas para entrenar la red
    batch : int
        Tamaño de batch
    metrics : list of str
        Metricas a imprimir por pantalla. Acepta cualquiera de las in-built de Keras.
    loss : str
        Funcion de coste o pérdida. Acepta cualquiera de las in-built de Keras.
    optim : str
        Optimizador del entrenamiento. Acepta cualquiera de las in-built de Keras.
    n_layer : int
        Número de capas de la red
    n_neuron : list of int
        Lista con las neuronas de cada capa
    alpha : float
        alpha utilizado para la activacion LeakyReLU
    model : tensorflow.keras.models.Model
        Modelo de red creado o importado
    history : DataFrame
        Historia de modelos previamente creados y ajustados
    method : str
        Metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced

    Methods
    -------
    create_model(hidden_activation="sigmoid", final_activation = 'sigmoid'):
        Crea un modelo de red neuronal con las activaciones ocultas y finales indicadas
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

    def __init__(self, n_layer, n_neuron, path_train,
                path_test, optim = 'adam', loss = 'binary_crossentropy',
                metrics = ['binary_accuracy'], epochs = 50, batch = 1,
                model = None, history = None, alpha=0.01, method=None):
        """
        Parameters
        ----------
        n_layer : int
            Número de capas de la red
        n_neuron : list of int
            Lista con las neuronas de cada capa
        alpha : float
            alpha utilizado para la activacion LeakyReLU
        path_train : str
            Ruta al archivo de datos de entrenamiento
        path_test : str
            Ruta al archivo de datos de prueba
        metrics : list of str
            Metricas a imprimir por pantalla. Acepta cualquiera de las in-built de Keras.
        loss : str
            Funcion de coste o pérdida. Acepta cualquiera de las in-built de Keras.
        optim : str
            Optimizador del entrenamiento. Acepta cualquiera de los in-built de Keras.
        model : tensorflow.keras.models.Model
            Modelo de red creado o importado de uno existente
        hist : DataFrame
            Historia de modelos previamente creados y ajustados
        method : str
            Metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced
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
        self.n_layer = n_layer
        self.n_neuron = n_neuron
        self.alpha = alpha          #alpha for LeakyRelu
        #self.res_size = res_size    #Size of the residuals

        #Leemos los datos
        self.data = ReadData(path_train, path_test)
        self.data.split(0.33)

        self.method = method
        if method != None:
            self.data.imbalanced(method=method)

        #Creamos modelo en caso de que ya hubiera uno y se lo estemos pasando
        self.model = model
        self.hist = history

    def create_model(self, hidden_activation = 'sigmoid', final_activation = 'sigmoid'):
        """
        Crea un modelo de perceptron basandose en los parámetros indicados

        Parameters
        ----------
        hidden_activation : str
            Activacion de las capas ocultas de la red. Acepta cualquiera de las in-built de Keras.
        final_activation : str
            Activacion de la capa de salida de la red. Acepta cualquiera de las in-built de Keras.
        """

        inputs = Input(shape=self.data.X_t.shape[1])

        first = True
        x = inputs

        # Tantas capas como se indiquen
        for l in range(0,self.n_layer):
            n = self.n_neuron[l]

            # Activacion indicada en los parámetros
            if hidden_activation=="lrelu":
                x = Dense(n)(x)                    #Capa plenamente conectada
                x = LeakyReLU(alpha=self.alpha)(x) #Le damos la activacion
            elif hidden_activation=="relu":
                x = Dense(n, activation="relu")(x)
            elif hidden_activation=="sigmoid":
                x = Dense(n, activation="sigmoid")(x)
            else:
                x = Dense(n, activation="tanh")(x)

        outputs = Dense(1, activation=final_activation)(x)
        self.model = Model(inputs = inputs, outputs = outputs, name = "MLP"
                                                                    + hidden_activation+'_'
                                                                    + final_activation+'_'
                                                                    + str(self.n_layer)+'_'+'-'
                                                                    +str(self.n_neuron).strip('[]').replace(', ','-')+'-'+'_'
                                                                    +str(self.epochs)+'_'
                                                                    +str(self.method))

    def compile(self):
        """
        Compila el modelo
        """
        self.model.compile(loss=self.loss, optimizer=self.optim)

    def train(self):
        """
        Entrena el modelo utilizando como datos de validacion los obtenidos en la division
        de los conjuntos en los datos
        """
        history = self.model.fit(self.data.X_t, self.data.y_t,
                    validation_data = (self.data.X_val, self.data.y_val),
                    epochs = self.epochs, batch_size = self.batch,
                    class_weight = self.data.class_weight)

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
            Array con las predicciones para el conjunto que fuera especificado
        """
        if test:
            prediction = self.model.predict(self.data.X_test)
        else: #Queremos predecir todo el conjunto de entrenamiento no solo el 66% de HO
            prediction = self.model.predict(self.data.X_train)

        return prediction

    def save_model(self):
        tf.keras.models.save_model(self.model,
                                    '../models/Perceptron/'+
                                    self.model.name+'.h5')
    def save_history(self):
        with open('../models/Perceptron/'+self.model.name, 'wb') as file_pi:
            pickle.dump(self.hist, file_pi)

    def get_labels_test(self):
        return self.data.y_test

    def get_labels_train(self):
        return self.data.y_train

    def get_n(self):
        return self.n_layer

    def get_history(self):
        return self.hist

    def get_optim(self):
        return self.optim

    def get_epochs(self):
        return self.epochs

    def get_loss(self):
        return self.loss

    def get_model(self):
        return self.model

    def get_name(self):
        return self.model.name
