import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# กำหนด path หลัก
PROJECT_PATH = r"C:\Vision_project"
DATA_DIR = "sign_language_data"

def load_training_data(base_dir):
    """
    โหลดข้อมูลสำหรับการเทรนโมเดล
    """
    X, y = [], []
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    print("เริ่มโหลดข้อมูล...")
    for letter in letters:
        letter_dir = os.path.join(base_dir, letter)
        files = [f for f in os.listdir(letter_dir) if f.endswith('_mask.png')]

        for filename in files:
            img_path = os.path.join(letter_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            X.append(img)
            y.append(letter)

    X = np.array(X).astype('float32') / 255.0  # ปรับขนาดค่าพิกเซล

    # เข้ารหัสป้ายเป็นตัวเลข
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # แปลงเป็นตัวเลข

    return X, y_encoded

def build_model(input_shape):
    """
    สร้างโมเดล CNN สำหรับการจำแนกประเภท
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))  # 9 classes for A-I

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    base_dir = os.path.join(PROJECT_PATH, DATA_DIR)
    X, y = load_training_data(base_dir)

    # แบ่งข้อมูลเป็นชุดเทรนและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ปรับขนาดข้อมูลเพื่อให้ตรงกับรูปแบบของโมเดล
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    # สร้างและเทรนโมเดล
    model = build_model((64, 64, 1))
    model.summary()

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # บันทึกโมเดล
    model.save(os.path.join(PROJECT_PATH, 'models', 'sign_language_model.h5'))
    print("โมเดลถูกบันทึกเรียบร้อยแล้ว")
