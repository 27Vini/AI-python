import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import *

df = pd.read_csv("sentiment140_processed.csv", encoding="ISO-8859-1")
df.drop("text", axis=1, inplace=True)
df = df.dropna(subset=["processed_text"])

X_train, X_, y_train, y_ = train_test_split(df["processed_text"], df["polarity"], test_size=0.40, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=42)

max_words = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

max_sequence_length = max(len(seq) for seq in tokenizer.texts_to_sequences(X_train))

X_train = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), padding="post", maxlen=max_sequence_length)
X_cv = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_cv), padding="post", maxlen=max_sequence_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), padding="post", maxlen=max_sequence_length)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length),
    tf.keras.layers.LSTM(units=64, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=True),
    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=3, activation="linear")
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_cv, y_cv), callbacks=[early_stopping])

train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

plot_life_line(train_loss, val_loss, train_accuracy, val_accuracy)

true_loss, true_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {true_loss},  Test Accuracy: {true_accuracy}")

model.save("modelo_sentimento.h5")
