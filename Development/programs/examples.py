from LSTM import LSTM_model
from SOM import SOM2
from NN import MLP_res, MLP
from plots import plot_net, plot_roc, plot_history, get_scores
import pickle
import tensorflow as tf
import sys

####################################################################################
#Ejemplo de creación y obtención de predicciones y graficos para el SOM
####################################################################################

som2 = SOM2([8,8],path_train='../Data/exoTrain.csv',path_test='../Data/exoTest.csv',
                method=None, type="gaussian")
som2.train(epochs=1)

pred = som2.predict(test=False) #Predict X_train
print("Train\n", get_scores(som2.get_labels_train(), pred))
plot_roc(som2,"Train", som2.get_labels_train(), pred, color='blue')
pred = som2.predict(test=True) #Predict X_test
print("Test\n", get_scores(som2.get_labels_test(), pred))
plot_roc(som2,"Test", som2.get_labels_test(), pred, color='blue', linestyle='--')

####################################################################################
#Ejemplo de creación y obtención de predicciones y graficos para la red LSTM
####################################################################################

lstm1 = LSTM_model(32, batch=128, path_train='../Data/exoTrain.csv',
                        path_test='../Data/exoTest.csv', epochs=1, method = "weights")

lstm1.create_model()
lstm1.compile()
lstm1.train()
lstm1.save_model()
lstm1.save_history()

pred = lstm1.predict(test=False) #Predict X_train
print("Train\n", get_scores(lstm1.get_labels_train(), pred))
plot_roc(lstm1,"Train", lstm1.get_labels_train(), pred, color='blue')
pred = lstm1.predict(test=True) #Predict X_test
print("Test\n", get_scores(lstm1.get_labels_test(), pred))
plot_roc(lstm1,"Test", lstm1.get_labels_test(), pred, color='blue', linestyle='--')
plot_history(lstm1)

####################################################################################
#Ejemplo de creación y obtención de predicciones y graficos para una red con residuales
####################################################################################

res1 = MLP_res(8, [64,64,64,64,64,64,64,64], path_train='../Data/exoTrain.csv',
                res_size=2, path_test='../Data/exoTest.csv', epochs=1, method = None)

res1.create_model(hidden_activation="relu")
res1.compile()
res1.train()
res1.save_model()
res1.save_history()
pred = res1.predict(test=False) #Predict X_train
print("Train\n", get_scores(res1.get_labels_train(), pred))
plot_roc(res1,"Train", res1.get_labels_train(), pred, color='blue')

pred = res1.predict(test=True) #Predict X_test
print("Test\n", get_scores(res1.get_labels_test(), pred))
plot_roc(res1,"Test", res1.get_labels_test(), pred, color='blue', linestyle='--')
plot_history(res1)

####################################################################################
#Ejemplo de creación y obtención de predicciones y graficos para un perceptron
####################################################################################

mlp1 = MLP(2, [64,64], path_train='../Data/exoTrain.csv',
            path_test='../Data/exoTest.csv', epochs=1, method = None)

mlp1.create_model(hidden_activation="relu")
mlp1.compile()
mlp1.train()
mlp1.save_model()
mlp1.save_history()
pred = mlp1.predict(test=False) #Predict X_train
print("Train\n", get_scores(mlp1.get_labels_train(), pred))
plot_roc(mlp1,"Train", mlp1.get_labels_train(), pred, color='blue')

pred = mlp1.predict(test=True) #Predict X_test
print("Test\n", get_scores(mlp1.get_labels_test(), pred))
plot_roc(mlp1,"Test", mlp1.get_labels_test(), pred, color='blue', linestyle='--')
plot_history(mlp1)

####################################################################################
# Ejemplo de carga de un modelo entrenado previamente
####################################################################################

mlp2 = tf.keras.models.load_model("../models/Perceptron/MLPsigmoid_sigmoid_4_-128-64-32-16-_10_FT.h5")

mlp2 = MLP(4, [128,64,32,16], path_train='../Data/exoTrain.csv',
            path_test='../Data/exoTest.csv', model=mlp2)

pred = mlp2.predict(test=False) #Predict X_train
print("Train\n", get_scores(mlp2.get_labels_train(), pred))
plot_roc(mlp2,"Train", mlp2.get_labels_train(), pred, color='blue')

pred = mlp2.predict(test=True) #Predict X_test
print("Test\n", get_scores(mlp2.get_labels_test(), pred))
plot_roc(mlp2,"Test", mlp2.get_labels_test(), pred, color='blue', linestyle='--')
