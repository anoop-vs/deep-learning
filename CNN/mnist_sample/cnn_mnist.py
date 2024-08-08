import tensorflow as tf
from numpy import argmax
import pandas as pd
import matplotlib.pyplot as plt


(train_x, y_train), (test_x, y_test) = tf.keras.datasets.mnist.load_data()


print("--------------------------------------\n")
print("Size of training Data Loaded:\n")
print('Train: X=%s, y=%s' % (train_x.shape, y_train.shape))
print('Test: X=%s, y=%s' % (test_x.shape, y_test.shape))
print("--------------------------------------\n")


# reshaping train and test sets 

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x .reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

#printing the shapes 
print("--------------------------------------\n")
print("Size of training Data Reshaped:\n")
print('Train: X=%s, y=%s' % (train_x.shape, y_train.shape))
print('Test: X=%s, y=%s' % (test_x.shape, y_test.shape))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

train_x = train_x.astype('float')/255
test_x = test_x.astype('float')/255
shape = train_x.shape[1:]
print("--------------------------------------\n")


print("CNN Model Creation \n")

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape= shape))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print("--------------------------------------\n")

model.summary()


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])

print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(train_x, y_train, epochs=5, batch_size = 128, validation_split = 0.2)
print("Training Finished.\n")
print("--------------------------------------\n")

# Plot and save accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('CNN/mnist_sample/results/mnist_accuracy_plot.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('CNN/mnist_sample/results/mnist_loss_plot.png')


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy= model.evaluate(test_x, y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Model Prediction.\n")
results = model.predict(test_x)
results = argmax(results,axis = 1)
results = pd.Series(results,name="Predicted Label")
submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
submission.to_csv("CNN/mnist_sample/results/MNIST-CNN.csv",index=False)
print("--------------------------------------\n")
