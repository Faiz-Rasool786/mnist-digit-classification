import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

unique,counts=np.unique(y_test, return_counts=True)
print("Test has",dict(zip(unique,counts)))

unique,counts=np.unique(y_train, return_counts=True)
print("Train has",dict(zip(unique,counts)))

indexes = np.random.randint(0,x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5,5,i+1)
    image=images[i]

    plt.imshow(image,cmap='gray')
    plt.axis('off')

plt.show()
plt.savefig("mnist-samples.png")
plt.close('all')

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

num_labels = y_train.shape[1]

print("x_train.shape =", x_train.shape)
print("Max pixel value in x_train[1] =", x_train[1].max())

image_size=x_train.shape[1]
input_size = image_size*image_size
x_train=np.reshape(x_train,[-1,input_size])
x_train=x_train.astype('float32')/255
x_test=np.reshape(x_test,[-1,input_size])
x_test=x_test.astype('float32')/255

batch_size=128
hidden_units=256
dropout=0.45

model=Sequential()

model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

plot_model(model, to_file='mlp.mnist.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20,batch_size=batch_size)
loss,acc = model.evaluate(x_test,y_test,batch_size=batch_size)

from tensorflow.keras.regularizers import l2
model.add(Dense(hidden_units, kernel_regularizer=l2(0.001), input_dim=input_size))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=batch_size)
loss,acc = model.evaluate(x_test,y_test,batch_size=batch_size)
