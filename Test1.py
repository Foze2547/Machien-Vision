import cv2
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# กำหนด path
#CSV_PATH = "D:\\Dataset_for_development\\test.csv"
#DATA_PATH = "D:\\Dataset_for_development\\Test"

DATA_PATH = r"D:\\Dataset_for_development\\Test Images"
CSV_PATH = r"D:\\Dataset_for_development\\test.csv"

# โหลด dataset
df = pd.read_csv(CSV_PATH)

# ฟังก์ชันการโหลดและเตรียมภาพ
def load_and_preprocess_image(image_name):
    img_path = os.path.join(DATA_PATH, image_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Cannot find the image: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # รีไซซ์ภาพให้เป็นขนาด 224x224
    img = img / 255.0  # Normalize ค่าให้เป็น 0-1
    return img

# ฟังก์ชันกำหนด custom metric/loss สำหรับโหลดโมเดล
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# ฟังก์ชันโหลดภาพจาก CSV และทำนายผล
def predict_images_from_csv():
    try:
        # โหลดโมเดลโดยระบุ custom_objects
        loaded_model = tf.keras.models.load_model('trained_model2.h5', custom_objects={"mse": mse})

        # ตรวจสอบว่ามีข้อมูลอย่างน้อย 1 แถว
        if len(df) < 1:
            raise ValueError("CSV file does not contain enough data.")

        # ฟังก์ชันสำหรับแสดงภาพแต่ละคู่
        class IndexTracker:
            def __init__(self, ax, df):
                self.ax = ax
                self.df = df
                self.index = 0
                self.total_images = len(df)

            def next_image(self, event):
                if self.index + 1 < self.total_images:
                    self.index += 1
                    self.update_plot()

            def prev_image(self, event):
                if self.index - 1 >= 0:
                    self.index -= 1
                    self.update_plot()

            def update_plot(self):
                row = self.df.iloc[self.index]
                img1 = load_and_preprocess_image(row['Image 1'])
                img2 = load_and_preprocess_image(row['Image 2'])

                # ทำนายผลการจำแนกประเภท
                pred_class = loaded_model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])

                # แสดงผลลัพธ์
                if pred_class[0][0] > 0.5:
                    winner_img = img2
                    loser_img = img1
                    winner_image_name = row['Image 2']
                    loser_image_name = row['Image 1']
                    result_text1 = "Image 2 "
                    result_text2 = "Image 1 "
                else:
                    winner_img = img1
                    loser_img = img2
                    winner_image_name = row['Image 1']
                    loser_image_name = row['Image 2']
                    result_text1 = "Image 1 "
                    result_text2 = "Image 2 "

                # แสดงภาพของอาหารที่ชนะและแพ้
                self.ax[0].imshow(winner_img)
                self.ax[0].set_title(f"Win: {result_text1}\n({winner_image_name})")
                self.ax[0].axis('off')

                self.ax[1].imshow(loser_img)
                self.ax[1].set_title(f"Loss: {result_text2}\n({loser_image_name})")
                self.ax[1].axis('off')

                plt.draw()

        # แสดงภาพ
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        tracker = IndexTracker(ax, df)

        # ปุ่ม "Next"
        ax_button_next = plt.axes([0.85, 0.01, 0.1, 0.075])
        button_next = Button(ax_button_next, 'Next')
        button_next.on_clicked(tracker.next_image)

        # ปุ่ม "Previous"
        ax_button_prev = plt.axes([0.05, 0.01, 0.1, 0.075])
        button_prev = Button(ax_button_prev, 'Previous')
        button_prev.on_clicked(tracker.prev_image)

        # แสดงภาพแรก
        tracker.update_plot()
        
        plt.show()

    except Exception as e:
        print("Error:", e)

# เรียกใช้งานฟังก์ชัน
predict_images_from_csv()
