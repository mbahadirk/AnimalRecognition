import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import timm
import os

# --- AYARLAR ---
model_path = "models/animal_classifier.pt"

class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison',
               'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach',
               'cow', 'coyote', 'crab', 'crow', 'deer', 'dog.jpg', 'dolphin', 'donkey',
               'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
               'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare',
               'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena',
               'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard',
               'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan',
               'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin',
               'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros',
               'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel',
               'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL YÜKLE ---
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# --- GÖRÜNTÜ DÖNÜŞÜMLERİ ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- TAHMİN FONKSİYONU ---
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_class = class_names[pred.item()]
        confidence = conf.item() * 100
        return predicted_class, confidence

# --- GÖRSEL YÜKLE VE TAHMİN YAP ---
def process_image(file_path):
    try:
        img = Image.open(file_path).resize((224, 224))
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img)
        image_label.image = tk_img

        pred_class, conf = predict_image(file_path)
        result_label.config(text=f"Tahmin: {pred_class}\nGüven: {conf:.2f}%")
    except Exception as e:
        result_label.config(text=f"Hata: {str(e)}")

# --- DOSYA SEÇİCİ KULLAN ---
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Görseller", "*.jpg *.jpeg *.png *.avif *.webp *.jpeg")])
    if file_path:
        process_image(file_path)

# --- SÜRÜKLE BIRAK OLAYI ---
def drop(event):
    file_path = event.data.strip("{}")  # boşluklu dosya isimleri için {}
    if os.path.isfile(file_path):
        process_image(file_path)

# --- TKINTER UI ---
window = TkinterDnD.Tk()
window.title("Hayvan Tanıma Uygulaması")
window.geometry("400x500")

title = tk.Label(window, text="Resmi Seçin veya Sürükleyin", font=("Arial", 14))
title.pack(pady=10)

image_label = tk.Label(window)
image_label.pack(pady=20)

result_label = tk.Label(window, text="Henüz bir görsel seçilmedi", font=("Arial", 12))
result_label.pack(pady=10)

select_button = tk.Button(window, text="Görsel Seç", command=load_image, font=("Arial", 12))
select_button.pack(pady=10)

# Sürükle bırak desteği
window.drop_target_register(DND_FILES)
window.dnd_bind("<<Drop>>", drop)

window.mainloop()
