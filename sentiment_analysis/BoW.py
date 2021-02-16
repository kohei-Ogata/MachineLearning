#文章や単語のカテゴリデータ（）は機械学習アルゴリズムに渡す前に数値に変換する
#Bag of Wordsモデルはテキストを数値の特徴量ベクトルとして表現できる

#文章全体から〇〇という一意なトークン（）からなる語彙を作成する
#各文書での各単語の出現回数を含んだ特徴量ベクトルを構築する

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

#3つの文章を疎な特徴量ベクトルに変換する
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

#vocablaryを覗くと、一意な単語と整数値を対応づけたものになっているらしい
#アルファベット順らしい

print(count.vocabulary_)

#さっきベクトルにしたものをみてみると、以下のように出る
#[[0 1 0 1 1 0 1 0 0]
#[0 1 0 0 0 1 1 0 1]
#[2 3 2 1 1 1 2 1 1]]
#ってなっているけど、左から数えて0番目、１番目、、と数えていくと、
#さっきcount.vocablary_で出した整数と単語が対応していて、その出現回数が出力されている
print(bag.toarray())

#ここからはTF-IDFなるものを使って単語の関連性の評価を行う
#TF{Term Frequency)=単語の出現頻度と、IDF(Inverse Document Frequency)=逆文書頻度の積で表す
#tf-idf
#で、このtfってのはわかるけど、idfってなんだ
#idfは以下の式で求められるらしい
#idf=log(nd/(1+df(d,t)))　らしい
#ndは文書の総数
#df(d,t)は単語tを含んでいる文書dの個数を表す
#分母に1足すのは、0を回避する
#logが使われているのは、過剰な重みを避けるため
#TfidfTransformerというもので、tfidfをなんか計算できるみたい

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#これからわかることは、最頻出単語であることは、有益な情報を含んでいるとは考えにくいということ(0.43)
#けど、この式はidfの計算方法がさっきのと少し違う
#さっきのは、logのなかの分子はndだけだったけど、それに+1してるし、tf-idfの式にもidfに+1されてる
#tf-idfを計算する前に、そもそもの出現頻度を正規化するのが普通だけどこの方法は結果をそのまま正規化してる
#ここはよくわからないけど、正規化されていない特徴量ベクトルvをL2正則化で割ることで、長さ1のベクトルが返される

#ちょっとわかった、長さ1のベクトル
#（このベクトルは単語の頻出回数とかに基づいたtf-idfの値で、色々な大きさがある）
#にするため、（=正規化：確率みたいにする感じで、マックスを1にする）
#そのベクトルの長さで割れば、絶対にマックスでも量が1になりそうな気がする
