from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib as plt
import math
import random
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from dataRead import ReadData
import sys
from scipy.spatial import distance


class neuron:
    """
    Clase que implementa los atributos y la funcionalidad de una neurona

    Attributes
    ----------
    weights : list of float/complex
        Vector de pesos de cada neurona
    label : int
        Etiqueta de clase de la neurona
    activation : float
        Valor de activacion de una neurona. En este caso será la distancia euclidea a la ganadora
    """
    def __init__(self, weights, label = None, activation = None):
        """
        Parameters
        ----------
        weights : list of float/complex
            Vector de pesos de cada neurona
        label : int
            Etiqueta de clase de la neurona
        activation : float
            Valor de activacion de una neurona. En este caso será la distancia euclidea a la ganadora
        """
        self.weights = weights
        self.label = label
        self.activation = activation

    def adjust_weights(self, alpha, x):
        """
        Realiza la actualizacion de los pesos de una neurona

        Parameters
        ----------
        alpha : float
            Factor de aprendizaje
        x : array-like
            Vector de entrada (observacion de los datos)
        """
        self.weights = self.weights + alpha * (x - self.weights)

class SOM2:
    """
    Clase que implementa el mapa auto-organizado

    Attributes
    ----------
    data : dataRead.ReadData
        Instancia de ReadData con los datos
    type : str
        Tipo de SOM, con vecindad gaussiana: "gaussian", o rectangular: "bubble"
    name : str
        Nombre del modelo
    map : list of list of SOM2.neuron
        Mapa bidimensional que contiene las neuronas
    """
    def __init__(self, map_size, path_train, path_test, type, method=None):
        """
        Parameters
        ----------
        type : str
            Tipo de SOM, con vecindad gaussiana: "gaussian", o rectangular: "bubble"
        path_train : str
            Ruta al archivo de datos de entrenamiento
        path_test : str
            Ruta al archivo de datos de prueba
        map_size : list
            Lista con los tamaños de ancho y alto del mapa
        method : str
            Metodo de tratamiento de datos. Acepta los presentes en dataRead.ReadData.imbalanced
        """
        self.data = ReadData(path_train, path_test)
        if method != None:
            self.data.imbalanced(method=method)

        self.type = type

        self.name = "SOM2_"+str(map_size[0])+"_"+str(method)+"_"+str(type)
        #Inicializamos el mapa con neuronas con pesos aleatorios
        self.map = [[neuron(np.asarray([np.random.rand(1) for i in range(self.data.X_train.shape[1])])) for i in range(map_size[0])] for j in range(map_size[1])]

    def activate(self, x, y):
        """
        Recibe una observacion y un mapa de neuronas y calcula la activacion
        distancia) de las neuronas del mapa

        Parameters
        ----------
        x : array-like
            Vector con la observacion
        y : int
            Valor binario con la clase de la observacion
        """

        for i in range(0,len(self.map)):
            for j in range(0,len(self.map[0])):
                # Solo para las de la misma clase o sin clase aun
                if (self.map[i][j].label==y) or (self.map[i][j].label==None):
                    # Vector de pesos
                    w = self.map[i][j].weights

                    # Distancia clasica
                    d = distance.euclidean(x,w)
                else:
                    d = sys.float_info.max #El peor posible
                self.map[i][j].activation = d

    def get_nearest(self):
        """
        Devuelve la neurona con la menor activación del mapa (ganadora)

        Returns
        -------
        list
            Lista con las coordenadas de la neurona que con la menor activacion
        """
        activations = np.asarray([[m.activation for m in row] for row in self.map])
        min_index = np.where(activations == np.amin(activations))
        return min_index

    def neuron_distance(self, win_i, win_j, i, j):
        """
        Calcula la distancia entre dos coordenadas: las de la neurona ganadora
        y las de cualquier otra neurona. Se realiza en un mapa con forma toroidal

        Parameters
        ----------
        win_i : int
            Posicion i de la neurona ganadora
        win_j : int
            Posicion j de la neurona ganadora
        i : int
            Posicion i de cualquier otra neurona
        j : int
            Posicion j de la neurona ganadora

        Returns
        -------
        float
            Valor de vecindad en base a las coordenadas
        """
        diff_i = abs(win_i - i)
        diff_j = abs(win_j - j)

        if (diff_i <= (len(self.map)-diff_i)):
            diff_i = diff_i
        else:
            diff_i = len(self.map)-diff_i
        if (diff_j <= (len(self.map[0])-diff_j)):
            diff_j = diff_j
        else:
            diff_j = len(self.map[0])-diff_j

        value = 2*( (diff_i*diff_i)/(len(self.map)*len(self.map)) + (diff_j*diff_j)/(len(self.map[0])*len(self.map[0])) )
        return value

    def adjust_neighbour_gauss(self, winner, x, l_rate, neigh_rate):
        """
        Ajusta los pesos de las neuronas con vecindad gaussiana

        Parameters
        ----------
        winner : array-like
            Coordenadas de la neurona ganadora
        x : array-like
            Observacion que provoco la ganadora
        l_rate : float
            Factor de aprendizaje dependiente de la epoca
        neigh_rate : float
            Factor de vecinadad dependiente de la epoca
        """
        for i in range(0,len(self.map)):
            for j in range(0,len(self.map[0])):
                distance = self.neuron_distance(winner[0][0], winner[1][0], i, j)
                # Factor de vecindad gaussiano
                neighbour_factor = math.exp(-distance / neigh_rate)
                if (self.data.y_train[i]==self.map[i][j].label) or (None==self.map[i][j].label):
                    self.map[i][j].adjust_weights(l_rate * neighbour_factor, x)

    def adjust_neightbour_bubble(self, winner, radio, alpha, x):
        """
        Ajusta los pesos de las neuronas con vecindad rectangular

        Parameters
        ----------
        winner : array-like
            Coordenadas de la neurona ganadora
        x : array-like
            Observacion que provoco la ganadora
        radio : float
            Factor de vecindad radial dependiente de la epoca
        alpha : float
            Factor de aprendizaje dependiente de la epoca
        x : array-like
            Observacion que provoco la ganadora
        """

        # Tamaño del radio de vecindad
        radioX = math.ceil(radio*len(self.map[0]))
        radioY = math.ceil(radio*len(self.map))

        radioX = math.ceil((radioX-1)/2)
        radioY = math.ceil((radioY-1)/2)

        i = winner[0][0]
        j = winner[1][0]

        #Ciclico e interconectado y toroidal, pero distancia de vecindad rectangular
        for l in range(j-radioY+1,j+radioY):
            for k in range(i-radioX+1,i+radioX):
                kk = k%len(self.map)
                ll = l%len(self.map[0])
                if (self.data.y_train[i]==self.map[ll][kk].label) or (None==self.map[ll][kk].label):
                    self.map[ll][kk].adjust_weights(alpha, x)


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
        preds = []
        if test:
            for i in range(0,self.data.X_test.shape[0]):

                x = self.data.X_test[i:i+1].transpose()
                d = sys.float_info.max
                # Busqueda de la neurona ganadora para una observacion
                for n1 in range(0,len(self.map)):
                    for n2 in range(0,len(self.map[0])):
                        d_i = distance.euclidean(x,self.map[n1][n2].weights)
                        if d_i < d:
                            d = d_i
                            w1 = n1
                            w2 = n2
                # Predecimos la clase de esa neurona
                preds.append(self.map[w1][w2].label)

            #Eliminamos las que no tienen etiqueta para los calculos de las medidas
            nones = [i for i in range(0,len(preds)) if preds[i]==None]
            self.data.y_test = np.delete(self.data.y_test, nones)

            for index in sorted(nones, reverse=True):
                del preds[index]

        else:
            for i in range(0,self.data.X_train.shape[0]):

                x = self.data.X_train[i:i+1].transpose()
                d = sys.float_info.max
                for n1 in range(0,len(self.map)):
                    for n2 in range(0,len(self.map[0])):
                        d_i = distance.euclidean(x,self.map[n1][n2].weights)
                        if d_i < d:
                            d = d_i
                            w1 = n1
                            w2 = n2
                preds.append(self.map[w1][w2].label)

            nones = [i for i in range(0,len(preds)) if preds[i]==None]
            self.data.y_train = np.delete(self.data.y_train, nones)

            for index in sorted(nones, reverse=True):
                del preds[index]

        return preds

    def train(self, epochs):
        """
        Entrena el mapa auto-organizado

        Parameters
        ----------
        epochs : int
            Numero de iteraciones que se realizara en el entrenamiento
        """

        print("Training...")

        # Ratios de vecindad
        learning_rate = 0.1
        neighbour_rate_init = 1

        # Bubble function
        alpha = 1
        radius_0 = len(self.map[0])

        for t in range(0, epochs):
            print("Epoch number: ", t)

            # Decrecimiento del factor de vecindad con las epocas
            neighbour_rate = neighbour_rate_init + t * neighbour_rate_init / epochs

            # Alimentamos el mapa en orden aleatorio
            indexes = [i for i in range(0,self.data.X_train.shape[0])]
            random.shuffle(indexes)

            # Decrecimiento del radio y factor de aprendizaje con las epocas
            radius = radius_0 * math.exp(-t/len(indexes))
            alpha = alpha/(1+t/len(indexes))
            #indexes = indexes[0:500] #Selecciono 1000 aleatoriamente
            error = 0

            x_i = 0
            for index in indexes:

                #Calculamos las activaciones para esa observacion
                x = self.data.X_train[index:index+1].transpose()
                self.activate(x, self.data.y_train[index])
                win = self.get_nearest()
                self.map[win[0][0]][win[1][0]].label = self.data.y_train[index]

                # Ajuste de pesos
                if self.type=="bubble":
                    self.adjust_neightbour_bubble(win, radius, alpha, self.data.X_train[index:index+1].transpose())
                elif self.type=="gaussian":
                    self.adjust_neighbour_gauss(win, self.data.X_train[index:index+1].transpose(), learning_rate, neighbour_rate)

                error = error + self.map[win[0][0]][win[1][0]].activation

                sys.stdout.write('%s\r' % str(round(100*x_i/len(indexes))))
                sys.stdout.write("%")
                sys.stdout.flush()
                x_i += 1

            print("\nEpoch score = ", error/len(indexes)) #Numero de instancias


    def get_labels_train(self):
        return self.data.y_train
    def get_labels_test(self):
        return self.data.y_test
    def get_name(self):
        return self.name
