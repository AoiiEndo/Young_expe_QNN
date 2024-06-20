# !pip uninstall qiskit
# !pip install qiskit
# !pip install tensorflow
# !pip install qiskit-machine-learning
# !pip install qiskit-aer

import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('young_quantum_data_observed5.csv')

decimal_places = 16  # 2桁の有効桁数を指定

for column in data.columns:
    data[column] = round(data[column].apply(lambda x: complex(x)), decimal_places)
for column in data.columns:
    data[column] = round(data[column].apply(lambda x: np.real(x)).astype(float), decimal_places)

# データをトレーニングとテストに分割
# 'Probability_0', 'Probability_1', 'Probability_2', 'Probability_3', 'Time_Step', 'Screen_Position', 'Wavelength', 'Slit_Width', 'Slit_Distance', 'distance_between_slits'
features = ['Quantum_State_0', 'Quantum_State_1', 'Quantum_State_2', 'Quantum_State_3', 'wavelength', 'Slit_Width', 'Slit_Distance', 'distance_between_slits']
y = data['observed_data']
X = data[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#学習する特徴量データの個数
input_data = 8
input_data = [input_data]

#結果として得られる項目数
X_train.shape[1]

# データを正規化
# X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
# X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# 量子回路の定義
num_qubits = X_train.shape[1]
num_classes = 2  # 0または1の2クラス分類

# パラメータ化された回路を作成
qc = QuantumCircuit(num_qubits)
params = np.linspace(0, 2 * np.pi, num_qubits)  # パラメータを均等に設定

# パラメータを設定
for i in range(num_qubits):
    qc.ry(params[i], i)

# 測定
qc.measure_all()

# バックエンドを選択
backend = qiskit.Aer.get_backend('qasm_simulator')

# トランスパイルしてアセンブル
compiled_circuit = transpile(qc, backend=backend)
qobj = assemble(compiled_circuit, shots=8192)

# シミュレーションを実行
job = backend.run(qobj)
result = job.result()
counts = result.get_counts()

from tensorflow.python.ops.gen_nn_ops import relu
# Kerasモデルの定義
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation=tf.sin, input_shape=input_data),
    tf.keras.layers.Dense(units=10, activation=tf.sin),
    tf.keras.layers.Dense(units=10, activation=tf.cos),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])
# モデルのコンパイル
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# モデルのトレーニング
history = model.fit(
  X_train, y_train,
  validation_data=(X_test, y_test),
  batch_size=100,
  epochs=100,
)
# epochsを大きくすればより0に近づくかも？同じ historyのとき干渉縞データを入れても０に近づく
print(history.history['loss'])
history_df = pd.DataFrame(history.history)
history_df['loss'].plot()

# データの評価#############################################################################
# 予測データを取得
predicted_values = model.predict(X_test)
# 予測データをファイルとして出力してより0-1に近い値を得られたモデルを求める。
df_pre = pd.DataFrame(predicted_values, columns=['1_10_10'])
# CSVファイルにデータを書き出す
csv_filename = "predicted_data3.csv"
df_pre.to_csv(csv_filename, index=False)
# データ数を生成（0からNまでの整数）
data_count = np.arange(len(y_test))
# モデルの各層の重みを取得
for layer in model.layers:
    if hasattr(layer, 'get_weights'):
        weights = layer.get_weights()
        if len(weights) > 0:
            print(f"Layer Name: {layer.name}")
            print("Weights:")
            print(weights)