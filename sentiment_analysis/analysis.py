#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:30:53 2021

@author: ogatakouhei
"""

import pyprind
import pandas as pd
import os
import numpy as np
import re
# basepath の値を展開した映画レビューデータセットのディレクトリに置き換える。
basepath = '/Users/ogatakouhei/Desktop/DataScience/machinLearning/sentiment_analysis/aclImdb'
labels = {'pos':1, 'neg':0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()

#testとtrainのpositive,negativeからそれぞれデータをとって,
#posなら1, negなら0にしてデータフレーム＝dfに入れてる？んじゃないかな？

for s in ('test','train'):
    for l in ('pos', 'neg'):
        #path名を結合している
        path = os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file), 'r', encoding = 'utf-8') as infile:
                txt = infile.read() 
            df = df.append([[txt,labels[l]]],ignore_index=True)
            pbar.update()
df.columns = ['review','sentiment']
print(df.columns)



#ランダムに配列、順番を設定し、訓練データにするのに役立つらしい
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
#ここで、dfからcsvに出力してる
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))
print(df.shape)

#これからBoWについて
#BoW.py参照

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)


#BoWの理解ができたら次に進む
#ここからはテキストデータのクレンジングを行う（綺麗にすること）
#しないけど、df.loc[0,review][-50:]で映画レビューデータセットの一つ目の文書から、
#最後の50行を取り出すと、HTML言語だったり、よくわからん物が出てくる
#だけど、:)みたいな顔文字は案外役立つから、そういった顔文字だけは残して、他を綺麗に削除する作業をする
#Pythonの正規表現reを使ってpreprocessor関数を作っていく
def preeprocessor(text):
    #ここでは<[^>]*>を使って、HTMLを完全に削除しようとしている
    text = re.sub('<[^>]*>','',text)
    #で、その後、もう少し複雑な正規表現を使って、顔文字を検索して、emotionに格納してるらしい
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    #で、格納したemoticonを処理済の末尾に付け足してるらしい
    #[\W]+を使って単語の一部ではない文字を全て削除して、小文字にもしているらしい
    text = (re.sub('[\W]+',' ', text.lower())+''.join(emoticons).replace('-',''))
    return text

#これでクレンジングは完了したっぽい
#最後にクレンジングしたテキストデータを繰り返し使うから、
#DataFrameオブジェクトに含まれている全てのレビューに適応する
df['review'] = df['review'].apply(preeprocessor)

#次に文書をトークン化する作業に入る
#トークンとは自然言語を解析する際、文書の最小単位として扱われる文字や文字列のこと
#らしく、トークン化とは、識別可能な言語単位にカットすること→意味のある言葉の最小単位？
#ワードステミングという手法が、トークン化する時に便利らしい
#ワードステミングとは、単語を原型に変換することで、関連する単語を同じ語幹にマッピングできるようにするプロセス
#Porterって人が、最初のアルゴリズムを開発したから、Porterステミングアルゴリズムとも呼ばれるらし
#で、そのPorterステミングはNLTK（Natural Language Toolkit for Python）ライブラリで実装されている
#conda install nltk
#nltk.pyで基礎的練習をしてみる
def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')


#大体やったら、次に

#次は、BoWモデルに基づいて、映画レビューを肯定的なレビューと否定的なレビューに分けるために、
#ロジスティック回帰モデルを訓練する
#クレンジングしたテキスト文書が含まれたDataFrameオブジェクトを
#25,000個の訓練用の文書と25,000個のテスト用の文書に分ける
x_train = df.loc[:25000,'review'].values
y_train = df.loc[:25000,'sentiment'].values
x_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values

#次にGridSearchオブジェクトを使ってロジスティック回帰モデルの最適なパラメーター集合を求める
#ここでは5分割交差検証を使う

#このあとはいよいよよくわからないコードが書かれてきた
#GridSearchオブジェクトと、交差検証をちゃんと理解できればわかるらしい
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range':[(1,1)],
               'vect__stop_words':[stop, None],
               'vect__tokenizer':[tokenizer, tokenizer_porter],
               'clf__penalty':['11','12']},
              {'vect__ngram_range':[(1,1)],
               'vect__stop_words':[stop, None],
               'vect__tokenizer':[tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty':['11','12'],
               'clf__C':[1.0, 10.0, 100.0]}]
Ir_tfidf = Pipeline([('vect','tfidf'),
                     ('clf',LogisticRegression(random_state=0,
                                               solver = 'liblinear'))])
gs_Ir_tfidf = GridSearchCV(Ir_tfidf, param_grid,
                           scorning='accuracy',
                           cv=5, verbose=2,
                           n_jobs=1)
gs_Ir_tfidf.fit(x_train,y_train)



print('Best parameter set: %s ' % gs_Ir_tfidf.best_params_)

print('CV Accuracy: %.3f' % gs_Ir_tfidf.best_score_)
crf = gs_Ir_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(x_test, y_test))


