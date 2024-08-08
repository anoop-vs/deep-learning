import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import asarray
from numpy import zeros
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def getGloveEmbeddings(glovefolderpath):
    print("---------------------- Getting Glove Embeddings -------------------------\n")
    embeddings_dictionary = dict()
    glove_file = open(f"{glovefolderpath}glove.6B.50d.txt", encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()
    print("----------------------  -------------------------\n")
    return embeddings_dictionary


glove_folder='REPLACE_WITH_GLOVE_FOLDER_PATH'
maxlen = 50
print("---------------------- Downloading Dataset -------------------------\n")

dataset =  pd.read_csv('https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection',sep='\t',names=['label','message'])
print("----------------------  -------------------------\n")


print("---------------------- Data PreProcessing -------------------------\n")
dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
X = dataset['message'].values
y = dataset['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
X_train = tokeniser.texts_to_sequences(X_train)
X_test = tokeniser.texts_to_sequences(X_test)
vocab_size = len(tokeniser.word_index) + 1
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)
print("----------------------  -------------------------\n")


embeddings_dictionary=getGloveEmbeddings(glove_folder)
embedding_matrix = zeros((vocab_size, maxlen))
for word, index in tokeniser.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
print("---------------------- Modelling -------------------------\n")

model=tf.keras.models.Sequential([
   tf.keras.layers.Embedding(input_dim=vocab_size,output_dim= maxlen, weights=[embedding_matrix], input_length=maxlen , trainable=False),
   tf.keras.layers.LSTM(maxlen),
   tf.keras.layers.Dense(1, activation='sigmoid')
])     
 
print(model.summary())
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)
print("----------------------  -------------------------\n")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("---------------------- Training -------------------------\n")

history=model.fit(x=X_train,
        y=y_train,
        epochs=50,
        callbacks=[early_stop],
        validation_split=0.2
         )

print("---------------------- Report -------------------------\n")


def c_report(y_true, y_pred):
   print("Classification Report")
   print(classification_report(y_true, y_pred))
   acc_sc = accuracy_score(y_true, y_pred)
   print(f"Accuracy : {str(round(acc_sc,2)*100)}")
   return acc_sc

def plot_confusion_matrix(y_true, y_pred):
   mtx = confusion_matrix(y_true, y_pred)
   sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, cmap="Blues", cbar=False)
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.savefig('LSTM/CustomLSTM/results/results.jpg')


preds = (model.predict(X_test) > 0.5).astype("int32")
c_report(y_test, preds)
plot_confusion_matrix(y_test, preds)
