# !pip install qulacs
# !pip3 install wurlitzer
# %load_ext wurlitzer
# !pip install qiskit
# !pip install qiskit-aer

from qulacs import QuantumState, QuantumCircuit
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
import pandas as pd
from google.colab import files
import random

# パラメータ設定 全てメートル表記 この辺は適当に
num_qubits = 2  # 2つの量子ビットを使用
num_samples = 10000  # データセットのサンプル数
num_time_steps = 10  # 時間ステップの数
screen_positions = np.linspace(-1, 1, num_samples)  # スクリーン上の位置を均等にサンプリング
slit_width = 2.0e-2  # スリットの幅
# wavelength = 7.5e-7  # 光の波長　赤色光
distance_between_slits = 10.0  # 第一スリットから第二スリットまでの距離
slit_to_slit_distance = 1.0e-2  # スリット間の距離
screen_to_slit_distance = 10.0  # スクリーンからスリットまでの距離

# スクリーンの位置と時間ステップごとの干渉データを保存するリスト
screen_positions_list = []
interference_data_list = []
time_list = []
observed_list = []  # 観測の有無を記録するリスト
quantum_states_list = []
wavelength_list = []

# 量子回路の構築（1回目のシミュレーションで量子状態を取得）
quantum_circuit = QuantumCircuit(num_qubits, num_qubits) #num_qubites個の量子ビットを0にして準備
quantum_circuit.cx(0, 1) #量子ビットが0,1のものをもつれさせる。
backend = Aer.get_backend('statevector_simulator') #回路を自作するためにbackendを設定。
result = execute(quantum_circuit, backend, shots=1).result()
quantum_state_initial = result.get_statevector()

# 一部観測あり ###########################################################################################################################################
# 各時間ステップでの干渉状態をシミュレーション
for t in range(num_time_steps):
    # 各時間ステップでのデータを保存するリスト
    interference_data = []
    quantum_states = []  # 量子状態を保存するリスト
    screen_positions = np.linspace(-1, 1, num_samples)

    # 各スクリーン位置で干渉状態をシミュレーション
    for screen_position in screen_positions:

        wavelength = random.uniform(3.6e-7, 8.3e-7)

        # 量子回路の構築
        circuit = QuantumCircuit(num_qubits, num_qubits)

        # スリット1から光を通す（位相の変化を考慮）
        phase_shift_slit1 = 2 * np.pi * screen_position / wavelength
        circuit.ry(phase_shift_slit1, 0)

        # スリット2から光を通す（位相の変化を考慮）
        phase_shift_slit2 = 2 * np.pi * (screen_position - slit_width) / wavelength
        circuit.ry(phase_shift_slit2, 1)

        # スリット2からの光が干渉する（CNOTゲートを使用）
        circuit.cx(0, 1)

        # スリット間の位相シフトを追加
        slit_phase_shift = 2 * np.pi * distance_between_slits * np.sin(slit_to_slit_distance / (2 * wavelength))
        circuit.ry(slit_phase_shift, 0)
        circuit.ry(slit_phase_shift, 1)

        # 観測の有無によってどちらのスリットから光が通過したかをシミュレート
        observed = 0
        if np.random.rand() < 0.5:
            observed = 1
            if np.random.rand() < 0.5:
                # 50%の確率でスリット1から
                circuit.measure(0, 0)
                circuit.measure(1, 1)
            else:
                # 50%の確率でスリット2から
                circuit.measure(0, 1)
                circuit.measure(1, 0)

        # シミュレーション実行
        result = execute(circuit, backend, shots=1).result()
        counts = result.get_counts()
        state_vector = result.get_statevector()

        # 干渉縞データに結果を追加
        interference_data.append(np.abs(state_vector) ** 2)  # 測定確率を追加

        # 量子状態をリストに追加
        quantum_states.append(state_vector)

        # 観測フラグを配列で記録
        observed_list.append(observed)

        # 波長を記録
        wavelength_list.append(wavelength)

    # スクリーンの位置データをリストに追加
    screen_positions_list.append(screen_positions)

    # 干渉データをリストに追加
    interference_data_list.append(interference_data)

    # 時間データをリストに追加
    time_list.append([t] * num_samples)

    # 量子状態データをリストに追加
    quantum_states_list.append(quantum_states)
######################################################################################################################################################
# 上か下どちらか一方を実行。
#全て観測あり
# 各時間ステップでの干渉状態をシミュレーション
for t in range(num_time_steps):
    # 各時間ステップでのデータを保存するリスト
    interference_data = []
    screen_positions = np.linspace(-1, 1, num_samples)

    # 各スクリーン位置で干渉状態をシミュレーション
    for screen_position in screen_positions:
        # 量子回路の構築
        circuit = QuantumCircuit(2, 2)  # 2つの量子ビットを使用

        # スリット1から光を通す（位相の変化を考慮）
        phase_shift_slit1 = 2 * np.pi * screen_position / wavelength
        circuit.ry(phase_shift_slit1, 0)  # 0番目の量子ビットにRYゲートを適用

        # スリット2から光を通す（位相の変化を考慮）
        phase_shift_slit2 = 2 * np.pi * (screen_position - slit_width) / wavelength
        circuit.ry(phase_shift_slit2, 1)  # 1番目の量子ビットにRYゲートを適用

        # スリット2からの光が干渉する（CNOTゲートを使用）
        circuit.cx(0, 1)

        # スリット間の位相シフトを追加
        slit_phase_shift = 2 * np.pi * distance_between_slits * np.sin(slit_to_slit_distance / (2 * wavelength))
        circuit.ry(slit_phase_shift, 0)
        circuit.ry(slit_phase_shift, 1)

         # 観測の有無によってどちらのスリットから光が通過したかをシミュレート
        if np.random.rand() < 0.5:
          # 50%の確率でスリット1から
          circuit.measure(0, 0)  # 0番目の量子ビットを観測
          circuit.measure(1, 1)  # 1番目の量子ビットを観測
          observed = 0  # 観測ありを示すフラグ (スリット1を通過)
        else:
          # 50%の確率でスリット2から
          circuit.measure(0, 1)  # 0番目の量子ビットを観測
          circuit.measure(1, 0)  # 1番目の量子ビットを観測
          observed = 1  # 観測ありを示すフラグ (スリット2を通過)


        # シミュレーション実行
        backend = Aer.get_backend('statevector_simulator')
        result = execute(circuit, backend, shots=1).result()
        counts = result.get_counts()
        state_vector = result.get_statevector()

        # 干渉縞データに結果を追加
        interference_data.append(np.abs(state_vector) ** 2)  # 測定確率を追加

        # 量子状態をリストに追加
        quantum_states_list.append(state_vector)

        # 観測フラグを配列で記録
        observed_list.append(observed)

    # スクリーンの位置データをリストに追加
    screen_positions_list.append(screen_positions)

    # 干渉データをリストに追加
    interference_data_list.append(interference_data)

    # 時間データをリストに追加
    time_list.append([t] * num_samples)
#############################################################################################################################################################
# データセットの作成
data = np.vstack(interference_data_list)
time_data = np.vstack(time_list)
position_data = np.vstack(screen_positions_list)
observed_data = np.vstack(observed_list)
quantum_data = np.vstack(quantum_states_list)
wavelength_data = np.vstack(wavelength_list)

# 前提条件のカラムデータを作成
# wavelength_column = np.full(num_samples, wavelength)
slit_width_column = np.full(num_samples, slit_width)
slit_distance_column = np.full(num_samples, slit_to_slit_distance)
distance_between_slits_column = np.full(num_samples, distance_between_slits)

# データセットの作成
# wavelength_data = np.vstack(wavelength_column)
slit_width_data = np.vstack(slit_width_column)
slit_distance_data = np.vstack(slit_distance_column)
distance_between_slits_data = np.vstack(distance_between_slits_column)

# データをPandas DataFrameに変換
df = pd.DataFrame(data, columns=[f'Probability_{i}' for i in range(4)])
df['Time_Step'] = np.repeat(np.arange(num_time_steps), num_samples)
df['Screen_Position'] = np.tile(screen_positions, num_time_steps)
df['observed_data'] = observed_data  # 観測の有無を追加
df['wavelength'] = wavelength_data
# df['Wavelength'] = np.repeat(wavelength_data, num_time_steps)  # 波長データ
df['Slit_Width'] = np.repeat(slit_width_data, num_time_steps)  # スリット幅データ
df['Slit_Distance'] = np.repeat(slit_distance_data, num_time_steps)  # スリット間距離データ
df['distance_between_slits'] = np.repeat(distance_between_slits_data, num_time_steps)  # スリット間距離データ

# 量子状態をデータフレームに追加
quantum_data_columns = [f'Quantum_State_{i}' for i in range(4)]
df[quantum_data_columns] = pd.DataFrame(quantum_data, columns=quantum_data_columns)

# CSVファイルにデータを書き出す
csv_filename = "young_quantum_data_observed5.csv"
df.to_csv(csv_filename, index=False)