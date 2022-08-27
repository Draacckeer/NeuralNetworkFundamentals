import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Entrenamiento finalizado.")


plt.xlabel('Epoca')
plt.ylabel('Error')
plt.plot(historial.history['loss'])
plt.show()

print("Predicting...")
prediccion = modelo.predict(np.array([[0.0], [22.0], [40.0]]))
print(prediccion)
