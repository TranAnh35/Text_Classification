import re
import joblib
import os
import pickle
import string
from unidecode import unidecode
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

def simple_preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def remove_accents(text):
    return unidecode(text)

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set([w.strip() for w in f.readlines()])

# Tải mô hình và các đối tượng cần thiết
model_path = 'model/'
data_path = 'data/'
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load(model_path + 'classes.npy')
tfidf_vect = pickle.load(open(model_path + "vectorizer.pickle", "rb"))
loaded_model = joblib.load(model_path + 'model.joblib')
stopwords = load_stopwords(data_path + 'vietnamese-stopwords-dash.txt')

def preprocess_text(text):
    lines = text
    lines = simple_preprocess(lines)
    lines = ViTokenizer.tokenize(lines)
    
    split_words = [x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\'') for x in lines.split()]
    lines = ' '.join([word for word in split_words if word not in stopwords])
    lines = remove_accents(lines)
    return lines

def detect_file_type(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    processed_text = preprocess_text(content)
    tfidf_x = tfidf_vect.transform([processed_text])
    prediction = loaded_model.predict(tfidf_x)
    return encoder.inverse_transform(prediction)[0]

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        display_file_type(file_path)

def display_file_type(file_path):
    file_type = detect_file_type(file_path)
    file_label.config(text=f"File Path: {file_path}")
    type_label.config(text=f"File Type: {file_type}")

def on_drop(event):
    file_path = event.data.strip('{}')
    if os.path.isfile(file_path):
        display_file_type(file_path)
    else:
        messagebox.showerror("Error", "Invalid file. Please drop a valid file.")

root = TkinterDnD.Tk()
root.title("File Type Detector")

file_label = tk.Label(root, text="File Path: ")
file_label.pack(pady=10)

type_label = tk.Label(root, text="File Type: ")
type_label.pack(pady=10)

browse_button = tk.Button(root, text="Browse", command=open_file_dialog)
browse_button.pack(pady=10)

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

root.geometry("400x200")
root.mainloop()
