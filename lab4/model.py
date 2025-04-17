def generate_model():
  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Dropout(0.2))

  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())

  model.add(Dropout(0.5))

  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dense(1024))
  model.add(Activation('relu'))

  model.add(Dense(10))
  model.add(Activation('softmax'))


  model.summary()

  adam = optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'], jit_compile=False)

  return model