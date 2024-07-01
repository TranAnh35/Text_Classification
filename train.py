import os
import sys

# Thêm thư mục hiện tại vào sys.path để có thể import các module khác
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process_data import *
from utils import create_classifier, save_model
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
import joblib
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/')
    parser.add_argument('--data_path', type=str, default='data/')
    
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path

    # Tạo mô hình Naive Bayes với Laplace Smoothing (alpha=1.0)
    model = create_classifier()

    # Tiền xử lý dữ liệu, lấy dữ liệu từ hàm preprocess_data trong process_data.py
    X_data, y_data, X_test, y_test = preprocess_data(data_path, model_path)
    
    # Chia dữ liệu thành tập train và tập validation
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.05, random_state=2024)
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán nhãn tập train, validation và test
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    # In ra độ chính xác trên tập train, validation và test
    print("Train accuracy: ", metrics.accuracy_score(y_train, train_predictions))
    print("Validation accuracy: ", metrics.accuracy_score(y_val, val_predictions))
    print("Test accuracy: ", metrics.accuracy_score(y_test, test_predictions))
    
    # Lưu mô hình vào đĩa
    save_model(model, model_path)
