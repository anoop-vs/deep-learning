import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy import argmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam


# load the dataset
print("Loading Dataset generation Intiated..")

path = 'https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/iris.csv'
df = pd.read_csv(path, header=None)

X=df.values[:,:-1]
y=df.values[:, -1]

X = X.astype('float')
y = LabelEncoder().fit_transform(y)


print("Train | Test Dataset generation Started..")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_features = X_train.shape[1]
output_class = 3

#Modelling a sample DNN
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(n_features,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(output_class,activation='softmax'))

# opt=Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# plot_model(model, 'DNN/model.png')

print("Training Started.")
history=model.fit(X_train, y_train, epochs=200, batch_size=16)
loss, acc = model.evaluate(X_test, y_test)
print("Training Finished.")

print(f'Test Accuracy:{round(acc*100)}')


# make a prediction
row = [9.1,7.4,5.4,6.2]
prediction = model.predict([row])
print('Predicted: %s (class=%d)' % (prediction, argmax(prediction)))




