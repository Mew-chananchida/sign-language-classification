import cv2
import numpy as np
import tensorflow as tf

# โหลดโมเดลที่บันทึกไว้
model_path = r"C:\Vision_project\models\sign_language_model.h5"
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    """
    ฟังก์ชันที่ใช้ในการปรับขนาดและ normalize ภาพ
    """
    img = cv2.resize(image, (64, 64)).astype('float32') / 255.0  # ปรับขนาดและ normalize
    img = img.reshape(1, 64, 64, 1)  # เปลี่ยนรูปร่างให้เหมาะกับโมเดล
    return img


def classify_sign(image):
    """
    ฟังก์ชันที่ใช้โมเดลเพื่อตรวจจับตัวอักษรจากภาพที่เตรียมไว้
    """
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # กำหนดคลาสตัวอักษรตามโมเดล
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    return classes[predicted_class]


def main():
    cap = cv2.VideoCapture(0)

    # กำหนดค่า ROI
    roi_top, roi_bottom, roi_left, roi_right = 100, 400, 100, 400

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ตัด ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        # แปลงเป็นภาพขาวดำเพื่อเน้นมือ
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        # ลด noise เพื่อเน้นการตรวจจับมือ
        kernel = np.ones((7, 7), np.uint8)  # ใช้ kernel ขนาดใหญ่ขึ้น
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # ตรวจจับ contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hand_contour = max(contours, key=cv2.contourArea)

            # วาด contour
            cv2.drawContours(roi, [hand_contour], -1, (0, 255, 0), 2)

            # ใช้โมเดลในการทำนาย
            x, y, w, h = cv2.boundingRect(hand_contour)
            hand_roi = mask[y:y + h, x:x + w]
            hand_roi = cv2.resize(hand_roi, (64, 64))  # ปรับขนาดให้ตรงกับโมเดล
            predicted_letter = classify_sign(hand_roi)

            # แสดงผลลัพธ์
            cv2.putText(frame, f"Letter: {predicted_letter}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # วาดกรอบ ROI
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

        # แสดงภาพ
        cv2.imshow('Sign Language Alphabet Detection', frame)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
