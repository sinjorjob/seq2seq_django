from django import forms


class UserForm(forms.Form):
     messages = forms.CharField(label='英文入力',max_length=500,
     min_length=1,widget=forms.Textarea(attrs=
     {'id': 'messages','placeholder':'ここに翻訳したい英文を入力してください\n'})
     )