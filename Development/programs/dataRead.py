from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from scipy.fft import fft, fftfreq

class ReadData:

    """
    Clase que implementa metodos de lectura y tratamiento de datos.
    Crea instancias de X_train y X_test, que se manipularan si es necesario
    para finalmente ser divididas en X_train, X_val (ambas de X_train) y X_test
    (de X_test)

    Attributes
    ----------
    train : DataFrame
        Conjunto con los datos originales de entrenamiento
    test: DataFrame
        Conjunto con los datos originales de prueba
    y_train: array-like
        Clase para el conjunto de entrenamiento
    y_test: array-like
        Clase para el conjunto de prueba
    X_train: array-like
        Conjunto de entrenamiento sin la clase
    X_test: array-like
        Conjunto de prueba sin la clase
    class_weight: dict of {int:int}
        Pesos que se asignarán a las instancias de diferentes clases

    Methods
    -------
    split(test_size = 0.33):
        Divide el conjunto de entrenamiento en entrenamiento y validación
    imbalanced(method):
        Trata el conjunto de datos mediante diversas técnicas
    adapt_features(n_features, n_t):
        Adapta el conjunto de entrenamiento y prueba en caso de querer usar
        características
    """

    def __init__(self, path_train, path_test):
        """
        Parameters
        ----------
        path_train : str
            Ruta al archivo de datos de entrenamiento
        path_test : str
            Ruta al archivo de datos de prueba
        """

        self.train = pd.read_csv(path_train)
        self.test = pd.read_csv(path_test)

        self.y_train = np.asarray(self.train.pop('LABEL'))
        self.X_train = np.asarray(self.train)

        self.y_test = np.asarray(self.test.pop('LABEL'))
        self.X_test = np.asarray(self.test)

        encoder = LabelEncoder()

        encoder.fit(self.y_train)
        self.y_train = encoder.transform(self.y_train)

        encoder.fit(self.y_test)
        self.y_test = encoder.transform(self.y_test)

        #Incializado en none para facilitar la creacion de modelos
        self.class_weight = None

    def adapt_features(self, n_features, n_t):
        """
        Adapta los datos para poder procesar varios "timesteps" (cada n_t) a la vez
        (varias "features") eliminando columnas del final

        Parameters
        ----------
        n_features: int
            Numero de características
        n_t : int
            Número de timesteps
        """

        while n_t % n_features != 0:
            n_t = n_t - 1

        #Acortamos los timesteps
        self.X_train = self.X_train[:,0:n_t]
        self.X_test = self.X_test[:,0:n_t]

    def split(self, test_size = 0.33):
        """
        Separa el conjunto de entrenamiento original en entrenamiento y validacion.
        Estratificado por clase.

        Parameters
        ----------
        test_size : float
            Tamaño que se quiere usar para el conjunto de validación

        """

        self.X_t, self.X_val, self.y_t, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                    stratify=self.y_train,
                                                                    test_size=test_size,
                                                                    random_state=42)


    def imbalanced(self, method):
        """
        Realiza varios metodos de tratamiento de clases imbalanceadas

        Parameters
        ----------
        method : str
            Identifica el tipo de tratamiento de datos que se van a aplicar.
            Sus valores pueden ser:
            - SMOTE: creacion de instancias de la clase minoritaria hasta el 25% de la clase mayoritaria
            - weights: asignación de pesos a las instancias en función de su clase
            - sub: submuestrea la clase mayoritaria hasta el numero de instancias de la minoritaria
            - FT: aplica la transformada de Fourier a los datos
        """
        neg, pos = np.bincount(self.y_train)
        total = neg + pos

        if method=="SMOTE":
            #Remuestrea la clase minoritaria hasta el 25% de la mayoritaria
            smote = SMOTE(sampling_strategy=0.25)
            self.X_train, self.y_train = smote.fit_sample(self.X_train, self.y_train)

        if method=="weights":
            weight_0 = (1 / neg)*(total)/2.0
            weight_1 = (1 / pos)*(total)/2.0
            #Deberán ser añadidos a la creacion del modelo
            self.class_weight = {0: weight_0, 1: weight_1}

        if method=="sub":
            # Submuestrea la clase mayoritaria
            index_pos = np.where(self.y_train==1)[0]
            index_neg = np.where(self.y_train==0)[0]

            #Muestra aleatoria
            samp = np.random.choice(index_neg, size=len(index_pos), replace=False)

            index = np.concatenate((index_pos, samp))

            self.y_train = self.y_train[index]
            self.X_train = self.X_train[index,:]

        if method=="FT":
            self.X_train = fft(self.X_train)
            self.X_test = fft(self.X_test)
