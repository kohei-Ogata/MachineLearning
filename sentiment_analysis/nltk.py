#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:15:00 2021

@author: ogatakouhei
"""
def tokenizer(text):
    return text.split()


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def torkenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(torkenizer_porter('runners like running and thus they run'))
#実行すると、下記のような結果が出る
#['runner', 'like', 'run', 'and', 'thu', 'they', 'run']
#running が run に変換されていることがわかる

#stop word removalというのが有効に使えるらしい
#ストップワードとは、isとか、shouldとか、ごくありふれた単語で
#tf-idfではなくて、単語の出現頻度とかを使う手法にはいいらしい
#tf-idfは、ごくありふれた単語は重みが下がるから、
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in torkenizer_porter('a runner likes running and runs a lot')[-10:]
     if w not in stop])