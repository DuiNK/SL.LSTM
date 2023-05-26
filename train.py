import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
model_save_path = 'modelABCDE.hdf5'
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data_v2')

# Actions that we try to detect
#actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
actions = np.array(['A', 'B', 'C', 'D', 'E'])
# Thirty videos worth of data
no_sequences = 80

# Videos are going to be 30 frames in length
sequence_length = 80

# Folder start
start_folder = 30
log_dir = os.path.join('Data_v2')
tb_callback = TensorBoard(log_dir=log_dir)

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(80, 258)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(actions.shape[0], activation='softmax'))

es_callback = tf.keras.callbacks.EarlyStopping(patience=100, verbose=1, monitor = 'categorical_accuracy')
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, batch_size=128, callbacks=[es_callback, cp_callback] )

# 保存したモデルのロード
model.save('action.h5')
# del model
# model.load_weights('action.h5')
#
# model = tf.keras.models.load_model(model_save_path)
