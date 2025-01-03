import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Пример данных (вопросы и ответы)
data = {
    "Привет": "Привет! Как я могу вам помочь?",
    "Как тебя зовут?": "Я простой чат-бот.",
    "Что ты умеешь?": "Я могу отвечать на простые вопросы.",
    "Пока": "До свидания!"
}

# Подготовка данных
inputs = list(data.keys())
outputs = list(data.values())

# Кодирование данных
encoder = LabelEncoder()
encoded_outputs = encoder.fit_transform(outputs)

# Создание токенов для входных данных
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(inputs)
encoded_inputs = tokenizer.texts_to_matrix(inputs, mode='binary')

# Параметры модели
input_dim = encoded_inputs.shape[1]
output_dim = len(set(encoded_outputs))

# Построение модели
model = Sequential([
    Dense(16, input_shape=(input_dim,), activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(output_dim, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(encoded_inputs, encoded_outputs, epochs=100, verbose=0)

# Функция для общения с ботом
def chat_bot_response(input_text):
    input_encoded = tokenizer.texts_to_matrix([input_text], mode='binary')
    prediction = model.predict(input_encoded)
    response_index = np.argmax(prediction)
    return encoder.inverse_transform([response_index])[0]

# Пример общения
while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["пока", "выход", "стоп"]:
        print("Бот: До свидания!")
        break
    response = chat_bot_response(user_input)
    print(f"Бот: {response}")
