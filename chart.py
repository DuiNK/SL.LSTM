import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

model_save_path = 'model.hdf5'
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Data')
model = load_model('model.hdf5')
# Actions that we try to detect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
log_dir = os.path.join('Data')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
y_test = np.argmax(y_test, axis=1)
y_train = np.argmax(y_train, axis=1)
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))


Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
Y_predt = model.predict(X_train)
y_predt = np.argmax(Y_predt, axis =1)
print_confusion_matrix(y_test, y_pred)
print_confusion_matrix(y_train, y_predt)

# # 推論専用のモデルとして保存
# model.save(model_save_path, include_optimizer=False)
#
# # モデルを変換(量子化)
# tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quantized_model = converter.convert()
#
# open(tflite_save_path, 'wb').write(tflite_quantized_model)
#
# interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
# interpreter.allocate_tensors()
#
# # 入出力テンソルを取得
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
#
