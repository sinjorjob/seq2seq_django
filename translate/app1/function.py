from app1.config import NUM_SAMPLES, S2S_MODEL, DECODER_MODEL, ENCODER_MODEL
import numpy as np
import tensorflow as tf
from keras.models import load_model

def load_text(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return lines


def get_char(lines):
    input_texts = []  #英語のデータ
    target_texts = [] #日本語のデータを格納
    input_characters = set()    #英文に使われている文字の種類
    target_characters = set()   #日本語に使われている文字の種類

    for line in lines[: min(NUM_SAMPLES, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # ターゲット分の開始をタブ「\t」で、終了を改行「\n」で表す。
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    #ソート処理
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    return input_texts, target_texts, input_characters, target_characters


def get_num_word(i_char, t_char, i_txt, t_txt):
    num_encoder_tokens = len(i_char)     #インプット（英語）の単語数
    num_decoder_tokens = len(t_char)    #アウトプット（日本語）の単語数
    max_encoder_seq_length = max([len(txt) for txt in i_txt]) #一番長い英文の数
    max_decoder_seq_length = max([len(txt) for txt in t_txt]) #一番長い日本語分の数
    return num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length


# 文章をone-hot表現に変換する関数
def sentence_to_vector(sentence, max_encoder_seq_length, num_encoder_tokens, input_token_index):
    vector = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for j, char in enumerate(sentence):
        vector[0][j][input_token_index[char]] = 1
    return vector


def load_models():
    model = load_model(S2S_MODEL, compile=False)
    encoder_model = load_model(ENCODER_MODEL)
    decoder_model = load_model(DECODER_MODEL)
    graph = tf.get_default_graph()
    return model, encoder_model, decoder_model, graph


def create_dict(i_chars, t_chars):
    #英語辞書データの生成
    input_token_index = dict([
        (char, i) for i, char in enumerate(i_chars)
        ])
    #日本語辞書データの生成
    target_token_index = dict([
        (char, i) for i, char in enumerate(t_chars)
        ])
    return input_token_index, target_token_index

#使用できない文字（コーパスにない文字）があったときに無効な文字を判定するために使う。
def is_invalid(message,i_chars):
    is_invalid =False
    for char in message:
        if char not in i_chars:
            is_invalid = True
    return is_invalid


def decode_sequence(input_seq, num_decoder_tokens, target_token_index, encoder_model,\
                   decoder_model, reverse_target_char_index, max_decoder_seq_length, graph):
    with graph.as_default():
        # 入力文(input_seq)を与えてencoderから内部状態を取得
        states_value = encoder_model.predict(input_seq)
    # 長さ1の空のターゲットシーケンスを生成
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # ターゲットシーケンスの最初の文字に開始文字であるタブ「\t」を入力
    target_seq[0, 0, target_token_index['\t']] = 1.
    # シーケンスのバッチのサンプリングループ
    # バッチサイズ1を想定
    stop_condition = False
    # 初期値として返答の文字列を空で作成。
    decoded_sentence = ''
    while not stop_condition:
        with graph.as_default():
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # トークンをサンプリングする        
        # argmaxで最大確率のトークンインデックス番号を取得
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # Index空文字を取得
        sampled_char = reverse_target_char_index[sampled_token_index]
        # 返答文字列にサンプリングされた文字を追加
        decoded_sentence += sampled_char
        # 終了条件：最大長に達するか停止文字を見つける。
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # ターゲット配列（長さ1）を更新
        # 長さ1の空のターゲットシーケンスを生成
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        #予測されたトークンの値を1にセットし次の時刻の入力にtarget_seqを使う。 
        target_seq[0, 0, sampled_token_index] = 1.
        # 内部状態を更新して次の時刻の入力に使う。
        states_value = [h, c]
    return decoded_sentence