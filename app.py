# 以下を「app.py」に書き込み

import streamlit as st
import numpy as np
import pandas as pd
import openjij as oj
import pyqubo
from openjij import SQASampler
from pyqubo import Array, Constraint
import math
from base64 import b64encode


st.title("「生徒のクラス分け」アプリ")
st.write("量子アニーリングマシン：OpenJij")

def process_uploaded_file(file):
    df, column1_data, column2_data, column3_data, column4_data = None, None, None, None, None
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(file)

        # 列ごとにデータをリストに格納
        column1_data = df.iloc[:, 0].tolist()
        column2_data = df.iloc[:, 1].tolist()
        column3_data = df.iloc[:, 2].tolist()
        column4_data = df.iloc[:, 3].tolist()

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    return df, column1_data, column2_data, column3_data, column4_data

def upload_file_youin():
    st.write("生徒の属性ファイルのアップロード")
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    if uploaded_file is not None:
        # アップロードされたファイルを処理
        with st.spinner("ファイルを処理中..."):
            df, column1_data, column2_data, column3_data, column4_data = process_uploaded_file(uploaded_file)

        # アップロードが成功しているか確認
        if df is not None:
            # アップロードされたCSVファイルの内容を表示
            st.write("アップロードされたCSVファイルの内容:")
            st.write(df)
            w=column1_data
            w1=column2_data
            w2=column3_data
            p=column4_data
            return w, w1, w2, p

def download_csv(data, filename='data.csv'):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=True)

    b64 = b64encode(csv.encode()).decode()
    st.markdown(f'''
    <a href="data:file/csv;base64,{b64}" download="{filename}">
        クラス分け結果のダウンロード
    </a>
    ''', unsafe_allow_html=True)



try:
    w, w1, w2, p = upload_file_youin()
    st.write("生徒数：N = ",len(w))
    N=len(w)
except Exception as e:
    # エラーが発生したときの処理
    st.error("CSVファイルをアップロード後に処理されます".format(e))
#w, w1, w2, p = upload_file_youin()

#生徒の数
#st.write(w)
#st.write("生徒数：N = ",len(w))
#N=len(w)
#print('生徒の数：'f'{N=}')

try:
# プルダウンメニューで1から15までの整数を選択
    selected_number = st.selectbox("クラス数を入力してください", list(range(1, 16)))

    # ボタンが押されるまで待機
    submit_button = st.button("確定したら押してください")

    # ボタンが押されたら以下のコードが実行される
    if submit_button:
        # 入力フィールドで選択された整数値を入力
        K = st.number_input("選択したクラス数：", min_value=1, max_value=15, value=selected_number)
except Exception as e:
    st.error("生徒数入力後に処理されます".format(e))


#p = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


#st.write(K)
# Find the number of unique elements in the list
#try:
#num_unique = len(set(p))
num_unique = K
# Create a zero matrix of size (length of list, number of unique elements)
one_hot = [[0 for _ in range(num_unique)] for _ in range(len(p))]

# For each element in the list, set the corresponding element in the one-hot matrix to 1
for i, element in enumerate(p):
    one_hot[i][element] = 1

p=np.array(one_hot)
# Print the one-hot matrix
st.write(p)

lam1 = 10
lam2 = 10
a=1
b=1
c=1
d=10

x = Array.create(name='x', shape=(N,K), vartype='BINARY')
cost1  = 1/K * sum((sum(w[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost2 = 1/K * sum((sum(w1[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w1[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost3 = 1/K * sum((sum(w2[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w2[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))

#同じクラスだと加算する
cost4_in=0
for k in range(K):
#  for i in range(N):
#    if p[i,k]==1:
#      cost4_in += (sum(p[i,k]*x[i,k] for k in range(K)))**2
    cost4_in = sum((sum(p[i, k] * x[i, k] for k in range(K)))**2 for i in range(N) if p[i, k] == 1)
cost4 = 1/N*cost4_in

cost = a*cost1 + b*cost2 + c*cost3 + d*cost4

penalty1 = lam1 * sum((sum(x[i,k] for k in range(K)) -1 )**2 for i in range(N))
penalty2 = lam2 * sum((sum(x[i,k] for i in range(N)) -N/K )**2 for k in range(K))
penalty = penalty1 + penalty2

y = cost + penalty
model = y.compile()
Q, offset = model.to_qubo()
#print(Q)

#シミュレーション実施
#from openjij import SQASampler
sampler = SQASampler()
sampleset = sampler.sample_qubo(Q, num_reads=1000)

#ndarrayに変換
sample_array = []
for i in range(N):
    sample_list = []
    for k in range(K):
        sample_list.append(sampleset.first.sample[f'x[{i}][{k}]'])
    sample_array.append(sample_list)
sample_array = np.array(sample_array)
df=sample_array

#結果表示
#st.write(sample_array)
# NumPy配列を表示
st.write("結果表示:")
st.write(sample_array)
#st.write(type(sample_array))


# ダウンロードボタンを表示
download_csv(sample_array)
st.write('')
st.write('')

#生徒の成績テーブルの平均
Wu = 1/K * sum(w[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('成績：'f'ave={Wu}')
W1u = 1/K * sum(w1[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('性別：'f'ave1={W1u}')
W2u = 1/K * sum(w2[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('要支援：'f'ave2={W2u}')
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')
st.write('')
#各クラスでの成績合計、コスト（分散）、標準偏差を表示
st.write('各クラスでの成績合計、コスト（分散）、標準偏差を表示')
cost = 0
for k in range(K):
    value = 0
    for i in range(N):
        value = value + sample_array[i][k] * w[i]
    st.write(f'{value=}')
    cost = cost + (value - Wu)**2
cost = 1/K * cost
st.write(f'{cost=}')
standard_deviation = math.sqrt(cost)#標準偏差
st.write(f'{standard_deviation=}')
st.write('')
#各クラスに対して置くべき生徒を表示
for k in range(K):
    st.write(f'{k=}', end=' : ')

    output_text = "    ".join([str(w[i]) for i in range(N) if sample_array[i][k] == 1])
    st.write(output_text)
#  for i in range(N):
#    if sample_array[i][k] == 1:
#        st.write(w[i], end=' ')
    st.write('')#改行
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')
#各クラスでの性別合計、コスト（分散）、標準偏差を表示
st.write('各クラスでの性別合計、コスト（分散）、標準偏差を表示')
cost1 = 0
for k in range(K):
    value1 = 0
    for i in range(N):
        value1 = value1 + sample_array[i][k] * w1[i]
    st.write(f'{value1=}')
    cost1 = cost1 + (value - W1u)**2
cost1 = 1/K * cost1
st.write(f'{cost1=}')
standard_deviation1 = math.sqrt(cost1)#標準偏差
st.write(f'{standard_deviation1=}')
st.write('')
#各クラスに対して置くべき生徒を表示
for k in range(K):
    st.write(f'{k=}', end=' : ')
    output_text = "    ".join([str(w1[i]) for i in range(N) if sample_array[i][k] == 1])
    st.write(output_text)
    st.write('')#改行
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')
#各クラスでの要支援合計、コスト（分散）、標準偏差を表示
st.write('各クラスでの要支援合計、コスト（分散）、標準偏差を表示')
cost2 = 0
for k in range(K):
    value2 = 0
    for i in range(N):
        value2 = value2 + sample_array[i][k] * w2[i]
    st.write(f'{value2=}')
    cost2 = cost2 + (value2 - W2u)**2
cost2 = 1/K * cost2
st.write(f'{cost2=}')
standard_deviation2 = math.sqrt(cost2)#標準偏差
st.write(f'{standard_deviation2=}')
st.write('')
#各クラスに対して置くべき生徒を表示
for k in range(K):
    st.write(f'{k=}', end=' : ')
    output_text = "    ".join([str(w2[i]) for i in range(N) if sample_array[i][k] == 1])
    st.write(output_text)
    st.write('')#改行
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')

#罰金項のチェック
st.write('生徒一人のクラスの確認：count', end='')
for i in range(N):
    count = 0
    for k in range(K):
        count = count + sample_array[i][k]
#  st.write(f'{count}', end=',')
output_text = "    ".join([str(count) for i in range(N)])
st.write(output_text)

#except Exception as e:
#    st.error("クラス数確定後に計算されます".format(e))
