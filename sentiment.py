#####
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import re

class Sentiment:
    def __init__(self, max_len=30):
        self.max_len = max_len
        self.vectorizer = joblib.load('IE_model.pkl')
        self.model = joblib.load('RF_model.pkl')
        self.tokenizer = self.get_tokenizer()
        self.stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

    def get_tokenizer(self):
        okt = Okt()
        return okt.morphs
    
    def tokenize(self, sentence):
        tok_sent = self.tokenizer(sentence, stem=True) # 토큰화
        sw_rm_sent = [word for word in tok_sent if word not in self.stopwords]
        return sw_rm_sent

    def analyze_sentiment(self, sentence):
        in_sent_fv = re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", sentence) # 한글과 공백을 제외한 문자 삭제 -> string
        in_sent_fv = self.tokenize(in_sent_fv) # 토큰화 및 불용어 제거
        in_sent_fv = self.vectorizer.texts_to_sequences([in_sent_fv])  # 정수 인코딩 -> 2차원 array로 만들어서 입력
        in_sent_fv = pad_sequences(in_sent_fv, maxlen = self.max_len) # 패딩
        result = self.model.predict(in_sent_fv)
        result = "긍정 리뷰입니다" if result > 0.5 else "부정 리뷰입니다"
        return result