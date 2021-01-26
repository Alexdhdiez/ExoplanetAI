# ExoplanetIA: Machine Learning for the detection of exoplanets

The Kepler mission was designed to study the stars of the near-Earth environment within the Milky Way, by analysing the light emitted by 150,000 stars, in an attempt to discover planets of similar size to the Earth orbiting their star within the habitable zone, where liquid water may exist. With this information, a search for extraterrestrial life could be conducted, as well as an estimate of potentially habitable planets. 

The aim of the project is to use Machine Learning techniques and specifically neural networks on the real data provided by the Kepler space satellite, with respect to the intensity of light (flux) recorded over more than 9 years, and to propose a neural network model to detect exoplanets by the Transit Method. 

## Author(s)

* Alejandro del Hierro Diez: developer
* Manuel Hermán Capitán and Alejandro Viloria Lanero: project tutors from HP | SCDS
* Benjamín Sahelices: tutor of the project from the School of Computer Engineering of the University of Valladolid

## Requirements and installation

The project software does not need to be installed. However, in order to use it, some Python libraries need to be installed using the command:

```
pip install -r requirements.txt
```
The file `requirements.txt` is located in the root of the project.

For the correct functioning of the programs we recommend the use of Python 3.7.

## Repository structure

From the root folder...
```
.
├── requirements.txt
├── Development
│   ├── Data
│   │   ├── exoTest.csv
│   │   └── exoTrain.csv
│   ├── models
│   │   ├── LSTM
│   │   │   ├── LSTM_16_32_100_weights
│   │   │   ├── LSTM_16_32_100_weights.h5
│   │   │   └── Muchos mas modelos de LSTM...
│   │   ├── Perceptron
│   │   │   ├── MLPlrelu_sigmoid_2_-128-64-_100_FT
│   │   │   ├── MLPlrelu_sigmoid_2_-128-64-_100_FT.h5
│   │   │   └── Muchos mas modelos de perceptron y residual...
│   │   └── images
│   │       ├── ROC_LSTM_16_1_60_None.png
│   │       └── Muchas mas imagenes sacadas de los modelos...
│   └── programs
│       ├── DataExploration.py
│       ├── LSTM.py
│       ├── NN.py
│       ├── SOM.py
│       ├── __init__.py
│       ├── __pycache__
│       │   └── Archivos fruto de ejecuciones de Python...
│       ├── dataRead.py
│       ├── examples.py
│       └── plots.py
├── Memoria
│   ├── images
│   │   └── imagenes utilizadas en la memoria...
│   └── MemoriaTFG.pdf
└── README.md
```
1. Under 'Development/programs' you will find the core of the project, with the files containing the necessary classes and methods. Examples of their use can be found in the file `.Development/programs/examples.py`.
2. Under `.Development/Data` you will find the files used in the project.
3. In `.Development/models` all the models that have been executed are saved, including all the tests, so not all of them are useful and they were simply saved to facilitate their recovery if necessary.
4. In `.Development/Memoria` you will find the project report, with its explanation and results. Also all the images used in it.
