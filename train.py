import os
import argparse
import joblib
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split  
from process_data import preprocess_data
from utils import save_model  
# Thêm thư mục hiện tại vào sys.path để có thể import các module khác
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Hàm tính toán xác suất tiên nghiệm P(class)
def calculate_class_priors(labels):
    class_counts = defaultdict(int)
    total_count = 0
    
    for label in labels:
        class_counts[label] += 1
        total_count += 1
    
    priors = {cls: count / total_count for cls, count in class_counts.items()}
    return priors

# Hàm tính toán xác suất có điều kiện P(word|class) với Laplace smoothing
def calculate_conditional_probabilities(X_train, y_train, alpha):
    class_counts = defaultdict(int)
    word_counts = defaultdict(lambda: defaultdict(int))
    vocabulary = set()
    
    for i in range(len(X_train)):
        document = X_train[i]
        label = y_train[i]
        class_counts[label] += 1
        
        for word in document.split():
            word_counts[label][word] += 1
            vocabulary.add(word)
    
    conditional_probs = defaultdict(lambda: defaultdict(float))
    for cls in class_counts:
        total_words_in_class = sum(word_counts[cls].values())
        total_words_in_vocabulary = len(vocabulary)
        
        for word in vocabulary:
            word_count = word_counts[cls][word]
            conditional_probs[cls][word] = (word_count + alpha) / (total_words_in_class + alpha * total_words_in_vocabulary)
    
    return conditional_probs

# Hàm huấn luyện mô hình Naive Bayes với Laplace smoothing
def train_naive_bayes(X_train, y_train, alpha=1.0):
    class_priors = calculate_class_priors(y_train)
    conditional_probs = calculate_conditional_probabilities(X_train, y_train, alpha)
    
    def predict(text):
        best_class = None
        best_score = float('-inf')
        
        for cls in class_priors:
            score = np.log(class_priors[cls])
            
            for word in text.split():
                if word in conditional_probs[cls]:
                    score += np.log(conditional_probs[cls][word])
                else:
                    score += np.log(alpha / (sum(conditional_probs[cls].values()) + alpha))
            
            if score > best_score:
                best_score = score
                best_class = cls
        
        return best_class
    
    return predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/')
    parser.add_argument('--data_path', type=str, default='data/')
    
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path

    # Tiền xử lý dữ liệu
    X_data, y_data, X_test, y_test = preprocess_data(data_path, model_path)
    
    # Chia dữ liệu thành tập train và tập validation
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.05, random_state=2024)
    
    # Huấn luyện mô hình Naive Bayes với Laplace smoothing
    model = train_naive_bayes(X_train, y_train, alpha=1.0)
    
    # Đánh giá mô hình trên tập train, validation và test
    train_predictions = [model(text) for text in X_train]
    val_predictions = [model(text) for text in X_val]
    test_predictions = [model(text) for text in X_test]

    # In ra độ chính xác trên tập train, validation và test
    train_accuracy = np.mean(train_predictions == y_train)
    val_accuracy = np.mean(val_predictions == y_val)
    test_accuracy = np.mean(test_predictions == y_test)
    
    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    # Lưu mô hình
    joblib.dump(model, model_path + 'model.joblib')
    print("Saved model to disk")
