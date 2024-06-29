import os
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy
from unidecode import unidecode

# Loại bỏ stopwords. Stopwords là những từ không mang ý nghĩa như "và", "là", "của",...
def remove_stopwords(data, stopwords):

    for i in range(len(data)):
        text = data[i] # Lấy dữ liệu ở dòng thứ i
        try:
            split_words =  [x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'') for x in text.split()] # Tách từ trong dữ liệu và chuyển về chữ thường
        except TypeError:
            split_words =  [] # Nếu không thể tách từ thì gán split_words là một list rỗng
        data[i] = ' '.join([word for word in split_words if word not in stopwords]) # Loại bỏ stopwords khỏi dữ liệu
        
    return data

def remove_accents(text):
    # Chuyển các ký tự có dấu thành không dấu
    return unidecode(text)

# Tiền xử lý dữ liệu
def preprocess_data(data_path, model_path):

    X_data = pickle.load(open(data_path + "x_train.pkl",'rb'))
    y_data = pickle.load(open(data_path + "y_train.pkl",'rb'))
    X_test = pickle.load(open(data_path + "x_test.pkl",'rb'))
    y_test = pickle.load(open(data_path + "y_test.pkl",'rb'))

    with open(data_path + 'vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
        stopwords = set([w.strip() for w in f.readlines()])

    X_data = remove_stopwords(X_data, stopwords)
    X_data = [remove_accents(text) for text in X_data]
    X_test = remove_stopwords(X_test, stopwords)
    X_test = [remove_accents(text) for text in X_test]

    # Vector hóa dữ liệu
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=10000)
    tfidf_vect.fit(X_data)
    # Lưu vectorizer để sử dụng sau này
    pickle.dump(tfidf_vect, open(model_path+"vectorizer.pickle", "wb"))
    tfidf_X_data =  tfidf_vect.transform(X_data)
    tfidf_X_test =  tfidf_vect.transform(X_test)

    # Mã hóa nhãn dữ liệu
    encoder = preprocessing.LabelEncoder()
    y_data_one_hot = encoder.fit_transform(y_data)
    y_test_one_hot = encoder.fit_transform(y_test)
    # Lưu encoder để sử dụng sau này
    numpy.save(model_path+'classes.npy', encoder.classes_)

    return tfidf_X_data, y_data_one_hot, tfidf_X_test, y_test_one_hot