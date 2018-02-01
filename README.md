# Keris
Simple keras-like deep learning framework built with numpy

### Installation
`pip install -U keris`

### Getting Started
Keris uses keras-like [functional API](https://keras.io/getting-started/functional-api-guide) to build graphs of layer. Here is simple example to build, train and save weights of a model.
```python
from keris.model import Model
from keris.layers import Input, Conv2D, LeakyReLU, MaxPool2D, Concatenate, Dropout, Dense
from keris.optimizers import Adam
from keris.losses import softmax_crossentropy
from keris.callbacks import EarlyStopping, ModelCheckpoint

input_layer = Input(input_shape=(3, 60, 60), name='input')
hidden = Conv2D(kernel_size=16, filters=(3, 3), stride=2, name='conv1')(input_layer)
hidden = LeakyReLU(name='leaky_relu1')(hidden)
hidden = shortcut = MaxPooling2D(pool_size=(2, 2), stride=2, name='max_pool1')(hidden)
hidden = Conv2D(kernel_size=32, filters=(3, 3), stride=2, padding='same', name='conv2')(hidden)
hidden = LeakyReLU(name='leaky_relu2')(hidden)
hidden = Concatenate(axis=0, name='concat1')([shortcut, hidden])
hidden = Dropout(rate=0.5, name='dropout1')(hidden)
hidden = Dense(units=100, name='dense1')(hidden)
hidden = LeakyReLU(name='leaky_relu3')(hidden)
hidden = Dense(units=10, name='dense2')(hidden)

optimizer = Adam(lr=1e-3, decay=0.9, beta1=0.7)
callbacks = [EarlyStopping(patience=5),
             ModelCheckpoint('example_keris', save_best_only=True)]


model = Model(hidden)
model.compile(loss_fn=softmax_crossentropy, optimizer=optimizer)

model.fit(train_data, validation_data, epochs=100, batch_size=32, callbacks=callbacks)

model.save('example_keris')
```

### About
[Keris/Kris](https://en.wikipedia.org/wiki/Kris) is a traditional weapon from Java. It's a dagger with wavy blade, mostly used for ritual and black magic. I built this for my skripsi (final year project for bachelor degree in Indonesia) and not allowed to use any framework except numpy/matplotlib.
