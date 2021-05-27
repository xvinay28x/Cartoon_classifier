from tensorflow import keras
train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                     rotation_range=40,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True,
                                                     fill_mode="nearest")


train_set = train.flow_from_directory("Image/",
                                     target_size=(200,200),
                                     batch_size=5,
                                     subset="training",
                                     class_mode="categorical")

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=16, kernel_size=3, input_shape=(200,200,3), activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=64, activation="relu"))
model.add(keras.layers.Dense(units=7, activation="softmax"))

model.compile(optimizer='adam',
              loss="poisson",
              metrics=['CategoricalAccuracy'])

model.fit(train_set,epochs=40)

model.save("cartoon_classifier.h5")
