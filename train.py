from process_data import *
from utils import *
import argparse


import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Tạo một đối tượng ArgumentParser
    parser.add_argument('--model_path', type=str, 
                        default='model/') # Thêm argument --model_path với kiểu dữ liệu là string

    parser.add_argument('--data_path', type=str, 
                        default='data/') # Thêm argument --data_path với kiểu dữ liệu là string và giá trị mặc định là 'data/'

    args = parser.parse_args() # Parse các argument từ command line
    model_path = args.model_path # Lấy đường dẫn đến thư mục chứa model
    data_path = args.data_path # Lấy đường dẫn đến thư mục chứa dữ liệu
    model = create_classifier() # Tạo mô hình lấy từ hàm create_classifier trong utils.py có thể chọn tên khác
    X_data, y_data, X_test, y_test = preprocess_data(data_path, model_path) # Tiền xử lý dữ liệu, lấy dữ liệu từ hàm preprocess_data trong process_data.py
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.05, random_state=2024) # Chia dữ liệu thành tập train và tập validation
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=512) # Huấn luyện mô hình
    
    train_predictions = model.predict(X_train) # Dự đoán nhãn tập train
    val_predictions = model.predict(X_val) # Dự đoán nhãn tập validation
    test_predictions = model.predict(X_test) # Dự đoán nhãn tập test
    
    val_predictions = val_predictions.argmax(axis=-1) # Lấy nhãn có xác suất cao nhất
    test_predictions = test_predictions.argmax(axis=-1) # Lấy nhãn có xác suất cao nhất
    train_predictions = train_predictions.argmax(axis=-1) # Lấy nhãn có xác suất cao nhất

    print("Train accuract", metrics.accuracy_score(train_predictions, y_train)) # In ra độ chính xác trên tập train
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val)) # In ra độ chính xác trên tập validation
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test)) # In ra độ chính xác trên tập test
    
    model_json = model.to_json() # Chuyển mô hình sang dạng json

    with open(model_path + "model.json", "w") as json_file:
        json_file.write(model_json) # Lưu mô hình dạng json
    
    # model.save_weights(model_path+"model.h5")
    model.save_weights(model_path + "model.weights.h5") # Lưu trọng số của mô hình
    print("Saved model to disk")
   

