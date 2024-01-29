import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split



df = pd.read_csv("sentiment140_processed.csv", encoding="ISO-8859-1")
df.drop("text",axis=1,inplace=True)
df = df.dropna(subset=["processed_text"])

X_train, X_, y_train, y_ = train_test_split(df["processed_text"], df["polarity"], test_size=0.30, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=42)

max_words = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), padding="post")
X_cv = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_cv), padding="post")
X_test = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), padding="post")

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=X_train.shape[1]),
    tf.keras.layers.LSTM(units=64,),
    tf.keras.layers.Dense(units=3, activation="linear")
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_cv, y_cv))

true_loss, true_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {true_loss},  Test Accuracy: {true_accuracy}")

model.save("modelo_sentimento.h5")