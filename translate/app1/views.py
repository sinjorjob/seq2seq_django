import numpy as np
import pickle
from django.http.response import HttpResponse
from django.shortcuts import render, render_to_response
from django.contrib.staticfiles.templatetags.staticfiles import static
from . import forms
from django.template.context_processors import csrf
from app1.config import NUM_SAMPLES, S2S_MODEL, ENCODER_MODEL, DECODER_MODEL, DATA_PATH
from app1.function import load_text, get_char, get_num_word, sentence_to_vector,\
     load_models, create_dict, decode_sequence, is_invalid
from keras.models import load_model

#モデル、GRAPHのロード
model, encoder_model, decoder_model, graph = load_models()
#コーパスファイルのロード
lines = load_text(DATA_PATH)   
#英語、日本語文章に分け、各単語の種類を取得
input_texts, target_texts, input_characters, target_characters = get_char(lines)
#インプット、アウトプットの単語数、最大長を取得
num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length = get_num_word\
(input_characters, target_characters, input_texts, target_texts)
#英語,日本語辞書データの生成
input_token_index, target_token_index = create_dict(input_characters, target_characters)
#逆引き辞書の生成
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# 応答用の辞書を組み立てて返す
def __makedic(k,txt):
    return { 'k':k,'txt':txt}

def translate(request):
    if request.method == 'POST':
        # テキストボックスに入力されたメッセージ
        input_seq = request.POST["messages"]
        if not is_invalid(input_seq, input_characters):
            form = forms.UserForm()
            vec = sentence_to_vector(input_seq, max_encoder_seq_length, num_encoder_tokens, input_token_index)
            rets = decode_sequence(vec,num_decoder_tokens,target_token_index,encoder_model,\
                   decoder_model, reverse_target_char_index,max_decoder_seq_length, graph)
            
            # 描画用リストに最新のメッセージを格納する
            talktxts = []
            talktxts.append(__makedic('b',input_seq))
            talktxts.append(__makedic('ai', rets))
            # 過去の応答履歴をセッションから取り出してリストに追記する
            saveh = []
            if 'hist' in request.session:
                hists = request.session['hist']
                saveh = hists
                for h in reversed(hists):
                    x = h.split(':')
                    talktxts.append(__makedic(x[0],x[1]))
            # 最新のメッセージを履歴に加えてセッションに保存する
            saveh.append('ai:' + rets)
            saveh.append('b:'+ input_seq)
            request.session['hist'] = saveh
            
            c = {
                 'form': form,
                 'rets':rets,
                 'talktxts':talktxts
                }

        else:
            err_message = "無効な文字列が含まれています。英文を入力してください。"
            form = forms.UserForm()
            c = {
                'form': form,
                'err_message':err_message,
                }
            c.update(csrf(request))
            return render(request,'app1/demo.html',c)


    else:
        # 初期表示の時にセッションもクリアする
        request.session.clear()
        # フォームの初期化
        form = forms.UserForm(label_suffix='：')

        c = {'form': form,
        }
    c.update(csrf(request))
    return render(request,'app1/demo.html',c)


   