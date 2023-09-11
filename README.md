# Pneumonia
Pneumonia identification model

Este es un modelo de redes convolucionales que analiza una base de imágenes de Radiografìas con y sin neumonìa.

El modelo se realizó con 3 capas ocultas y una capa de salida 

# Crear el modelo CNN
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(1, activation='sigmoid'))


Se creo un metodo para escalar las imagenes a un tamaño comun, lo que redujo significativamente los tiempos de procesamiento y las irregularidades del dataset.


Se hizo un tunning, pero los resultados arrojados fueron una desmejora con respecto al primer modelo probado:
Segun el tuning:
Mejores hiperparámetros encontrados: {'optimizer': 'sgd', 'dropout_rate': 0.4}
Exactitud del mejor modelo en el conjunto de prueba: 0.5641025641025641

Mientras que el modelo original termino con una accuracy: 0.8995

El original fue:
-------------------------
# Genera el modelo de REDES NEURONALES CONVOLUCIONALES
model = Sequential()

# Capa convolucional 1

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa convolucional 2

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa convolucional 3

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Capa completamente conectada

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))



--------------------------------



La precision del modelo es de 89.9%

