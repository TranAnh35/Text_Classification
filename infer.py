# thư viện NLP tiếng Việt
import re
import joblib
from pyvi import ViTokenizer, ViPosTagger

from tqdm import tqdm
import numpy as np

# Thư viện khác
import argparse
import os 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from unidecode import unidecode
import warnings
warnings.filterwarnings("ignore")

def simple_preprocess(text):
    # Xóa các ký tự không cần thiết và chuyển về chữ thường
    text = re.sub(r'\W', ' ', text) # Thay thế ký tự không phải chữ cái, số, hoặc dấu cách bằng dấu cách
    text = re.sub(r'\s+', ' ', text) # Thay thế nhiều dấu cách bằng một dấu cách
    text = text.lower() # Chuyển về chữ thường
    return text

def remove_accents(text):
    # Chuyển các ký tự có dấu thành không dấu
    return unidecode(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='model/')
    parser.add_argument('--data_path', type=str,
                        default='data/')
    parser.add_argument('--text', type=str,
                        default="aaa")
    args = parser.parse_args() # Parse các argument từ command line
    lines = args.text # Lấy dữ liệu train
    model_path = args.model_path # Lấy đường dẫn đến thư mục chứa dữ liệu
    data_path = args.data_path # Lấy đường dẫn đến thư mục chứa dữ liệu
    lines = lines.splitlines() # Tách dữ liệu thành từng dòng
    lines = ' '.join(lines) # Chuyển list các dòng thành một chuỗi
    lines = simple_preprocess(lines) # Tiền xử lý dữ liệu
    lines = ViTokenizer.tokenize(lines) # Tokenize dữ liệu

    # Chỉ định mã hóa UTF-8 khi đọc tệp
    with open(data_path+'vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
        stopwords = set([w.strip() for w in f.readlines()])

    try:
        split_words = [x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'') for x in lines.split()]
    except TypeError:
        split_words = []
    lines = ' '.join([word for word in split_words if word not in stopwords])
    x = [lines]
    x = [remove_accents(text) for text in x]

    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load(model_path+'classes.npy')

    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
    tfidf_vect = pickle.load(open(model_path+"vectorizer.pickle", "rb"))

    tfidf_x = tfidf_vect.transform(x)

    loaded_model = joblib.load(model_path + 'model.joblib')
    print("-----------------------------------------------------------------------------------------------")
    print(encoder.inverse_transform(loaded_model.predict(tfidf_x))[0])
    print("-----------------------------------------------------------------------------------------------")
