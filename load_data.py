# from pyvi import ViTokenizer
# from tqdm import tqdm
# import pickle
# import os
# import argparse
# import warnings
# import re
# from unidecode import unidecode

# warnings.filterwarnings("ignore")

# def simple_preprocess(text):
#     # Xóa các ký tự không cần thiết và chuyển về chữ thường
#     text = re.sub(r'\W', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = text.lower()
#     return text

# def remove_accents(text):
#     # Chuyển các ký tự có dấu thành không dấu
#     return unidecode(text)

# def get_data(folder_path):
#     data = []
#     labels = []
#     dirs = os.listdir(folder_path)
#     for path in tqdm(dirs):
#         dir_path = os.path.join(folder_path, path)
#         if os.path.isdir(dir_path):
#             file_paths = os.listdir(dir_path)
#             for file_path in file_paths:
#                 full_file_path = os.path.join(dir_path, file_path)
#                 try:
#                     with open(full_file_path, 'r', encoding="utf-16") as f:
#                         lines = f.readlines()
#                         lines = ' '.join(lines)
#                         lines = simple_preprocess(lines)
#                         lines = remove_accents(lines)
#                         lines = ViTokenizer.tokenize(lines)
#                         data.append(lines)
#                         labels.append(path)
#                 except Exception as e:
#                     print(f"Error processing file {full_file_path}: {e}")
#     return data, labels

# def save_data(data, labels, folder_path, filename_prefix):
#     try:
#         with open(os.path.join(folder_path, f'{filename_prefix}_train.pkl'), 'wb') as f:
#             pickle.dump(data, f)
#         with open(os.path.join(folder_path, f'{filename_prefix}_test.pkl'), 'wb') as f:
#             pickle.dump(labels, f)
#         print(f"Saved {filename_prefix} data and labels to {folder_path}")
#     except Exception as e:
#         print(f"Error saving {filename_prefix} data: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='data/')
#     args = parser.parse_args()
#     folder_path = args.data_path
    
#     X_data, y_data = get_data(os.path.join(folder_path, 'Train_Full'))
#     save_data(X_data, y_data, folder_path, 'x')

#     X_test, y_test = get_data(os.path.join(folder_path, 'Test_Full'))
#     save_data(X_test, y_test, folder_path, 'y')


from pyvi import ViTokenizer
from tqdm import tqdm
import pickle
import os
import argparse
import warnings
import re
from unidecode import unidecode

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

def get_data(folder_path):
    data = []
    labels = []
    dirs = os.listdir(folder_path) # Lấy danh sách các thư mục con trong thư mục folder_path
    for path in tqdm(dirs): # Duyệt qua từng thư mục con
        dir_path = os.path.join(folder_path, path) # Đường dẫn đến thư mục con
        if os.path.isdir(dir_path): # Kiểm tra xem có phải thư mục không
            file_paths = os.listdir(dir_path) # Lấy danh sách các file trong thư mục con
            # file_paths = file_paths[:2000]
            for file_path in file_paths: # Duyệt qua từng file
                full_file_path = os.path.join(dir_path, file_path) # Đường dẫn đến file
                try:
                    with open(full_file_path, 'r', encoding="utf-16") as f: # Mở file
                        lines = f.readlines() # Đọc nội dung file
                        lines = ' '.join(lines) # Chuyển list các dòng thành một chuỗi
                        lines = simple_preprocess(lines) # Tiền xử lý dữ liệu
                        lines = remove_accents(lines) # Tiền xử lý dữ liệu
                        lines = ViTokenizer.tokenize(lines) # Tokenize dữ liệu
                        data.append(lines) # Thêm dữ liệu vào list data
                        labels.append(path) # Thêm nhãn vào list labels
                except Exception as e:
                    print(f"Error processing file {full_file_path}: {e}")
    return data, labels

def save_data(data, labels, folder_path, filename_prefix):
    try:
        with open(os.path.join(folder_path, f'x_{filename_prefix}.pkl'), 'wb') as f: # Mở file để ghi dữ liệu dưới dạng binary
            pickle.dump(data, f) # Ghi dữ liệu vào file
        with open(os.path.join(folder_path, f'y_{filename_prefix}.pkl'), 'wb') as f: # Mở file để ghi dữ liệu dưới dạng binary
            pickle.dump(labels, f) # Ghi dữ liệu vào file
    except Exception as e:
        print(f"Error saving {filename_prefix} data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Tạo một đối tượng ArgumentParser
    parser.add_argument('--data_path', type=str, default='data') # Thêm argument --data_path với giá trị mặc định là 'data'
    args = parser.parse_args() # Parse các argument từ command line
    folder_path = args.data_path # Lấy đường dẫn đến thư mục chứa dữ liệu
    
    train_folder = os.path.join(folder_path, 'Train_Full') # Đường dẫn đến thư mục chứa dữ liệu train
    test_folder = os.path.join(folder_path, 'Test_Full') # Đường dẫn đến thư mục chứa dữ liệu test
    
    X_data, y_data = get_data(train_folder) # Lấy dữ liệu train
    save_data(X_data, y_data, folder_path, 'train') # Lưu dữ liệu train
    
    X_test, y_test = get_data(test_folder) # Lấy dữ liệu test
    save_data(X_test, y_test, folder_path, 'test') # Lưu dữ liệu test