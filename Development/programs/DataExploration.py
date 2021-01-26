import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

####################################################################################
# Archivo con el que se generaron gráficos de las estrellas y de las transformadas de fourier
####################################################################################

# Lectura de los datos
data = pd.read_csv('../Data/exoTrain.csv')
y = np.asarray(data.pop('LABEL'))
x = np.asarray(data)

star_2 = np.asarray(data.iloc[1,1:]) #Exoplaneta

star_1 = np.asarray(data.iloc[-1,1:]) #No exoplaneta
plt.figure(figsize=(15,5))
plt.plot(np.array(range(3196)), star_2, label="Confirmed exoplanet")
plt.plot(np.array(range(3196)), star_1, label="Unconfirmed exoplanet")
plt.legend()
plt.title("Flux curve")
plt.xlabel("Flux time")
plt.ylabel("Flux value")
plt.show()

t = len(star_1)

#Hacemos transformada de fourier
fft1 = fft(star_1)
fft2 = fft(star_2)

freq = fftfreq(t)

plt.figure()
plt.subplot(211)
plt.plot(freq, fft1.real**2 + fft1.imag**2, color="orange")
plt.title("Unconfirmed exoplanet")
plt.subplot(212)
plt.plot(freq, fft1.real**2 + fft2.imag**2)
plt.title("Confirmed exoplanet")
plt.show()
