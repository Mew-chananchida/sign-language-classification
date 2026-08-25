import cv2
import numpy as np
import os
from datetime import datetime

# กำหนด path หลัก
PROJECT_PATH = r"C:\Vision_project"
DATA_DIR = "sign_language_data"


def create_data_directories():
    """
    สร้างโครงสร้างโฟลเดอร์สำหรับเก็บข้อมูลการtrain
    """
    # สร้างโฟลเดอร์หลักของโปรเจค
    if not os.path.exists(PROJECT_PATH):
        os.makedirs(PROJECT_PATH)

    # สร้างโฟลเดอร์สำหรับเก็บข้อมูล
    base_dir = os.path.join(PROJECT_PATH, DATA_DIR)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # สร้างโฟลเดอร์ย่อยสำหรับแต่ละตัวอักษร A-I
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for letter in letters:
        letter_dir = os.path.join(base_dir, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)

    # สร้างโฟลเดอร์สำหรับเก็บโมเดลที่เทรนแล้ว
    models_dir = os.path.join(PROJECT_PATH, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    return base_dir


def collect_training_data():
    """
    เก็บข้อมูลภาพสำหรับการฝึกฝนโมเดล
    """
    cap = cv2.VideoCapture(0)
    base_dir = create_data_directories()

    # กำหนดค่า ROI
    roi_top = 100
    roi_bottom = 400
    roi_left = 100
    roi_right = 400

    current_letter = 'A'
    capture_mode = False
    frame_count = 0

    # สร้างไฟล์ log
    log_file = os.path.join(PROJECT_PATH, 'data_collection_log.txt')

    print("=== โปรแกรมเก็บข้อมูลภาษามือ ===")
    print(f"เก็บข้อมูลที่: {base_dir}")
    print("คำสั่ง:")
    print("  's' - เริ่ม/หยุดการเก็บข้อมูล")
    print("  'n' - เปลี่ยนไปตัวอักษรถัดไป")
    print("  'q' - ออกจากโปรแกรม")
    print("  'r' - รีเซ็ตการนับเฟรมปัจจุบัน")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ตัด ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        # แปลงเป็น HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ปรับค่าสีผิว
        lower_skin = np.array([0, 25, 80], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # ปรับปรุงภาพ
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # แสดงสถานะบนหน้าจอ
        status_color = (0, 255, 0) if capture_mode else (0, 0, 255)
        cv2.putText(frame, f"Current Letter: {current_letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Capture Mode: {'ON' if capture_mode else 'OFF'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Frames Captured: {frame_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # วาดกรอบ ROI
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), status_color, 2)

        # บันทึกภาพเมื่ออยู่ในโหมดเก็บข้อมูล
        if capture_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            letter_dir = os.path.join(base_dir, current_letter)

            # บันทึกทั้งภาพ ROI และ mask
            roi_filename = os.path.join(letter_dir, f"{current_letter}_{timestamp}_roi.png")
            mask_filename = os.path.join(letter_dir, f"{current_letter}_{timestamp}_mask.png")

            cv2.imwrite(roi_filename, roi)
            cv2.imwrite(mask_filename, mask)
            frame_count += 1

            # บันทึก log
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp},{current_letter},{roi_filename},{mask_filename}\n")

            # หน่วงเวลาเล็กน้อย
            cv2.waitKey(100)

        # แสดงภาพ
        cv2.imshow('Data Collection - Sign Language', frame)
        cv2.imshow('Mask', mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nจบการทำงาน")
            print(f"จำนวนภาพทั้งหมดที่เก็บได้: {frame_count}")
            break
        elif key == ord('s'):
            capture_mode = not capture_mode
            if capture_mode:
                print(f"\nเริ่มเก็บข้อมูลสำหรับตัวอักษร {current_letter}")
            else:
                print(f"หยุดเก็บข้อมูล - เก็บได้ {frame_count} เฟรม")
        elif key == ord('n'):
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            current_index = letters.index(current_letter)
            current_letter = letters[(current_index + 1) % len(letters)]
            print(f"\nเปลี่ยนไปเก็บข้อมูลตัวอักษร {current_letter}")
            frame_count = 0
        elif key == ord('r'):
            frame_count = 0
            print("\nรีเซ็ตการนับเฟรม")

    cap.release()
    cv2.destroyAllWindows()


def prepare_training_data(base_dir=None):
    """
    เตรียมข้อมูลสำหรับการเทรนโมเดล
    """
    if base_dir is None:
        base_dir = os.path.join(PROJECT_PATH, DATA_DIR)

    X = []  # ข้อมูลภาพ
    y = []  # ลาเบล
    file_paths = []  # เก็บ path ของไฟล์

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    print("เริ่มโหลดข้อมูล...")
    for letter in letters:
        letter_dir = os.path.join(base_dir, letter)
        if not os.path.exists(letter_dir):
            print(f"ไม่พบโฟลเดอร์สำหรับตัวอักษร {letter}")
            continue

        files = [f for f in os.listdir(letter_dir) if f.endswith('_mask.png')]
        print(f"พบข้อมูลตัวอักษร {letter}: {len(files)} ไฟล์")

        for filename in files:
            img_path = os.path.join(letter_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (64, 64))
                X.append(img)
                y.append(letter)
                file_paths.append(img_path)

    print(f"\nโหลดข้อมูลเสร็จสิ้น")
    print(f"จำนวนข้อมูลทั้งหมด: {len(X)} ภาพ")

    # บันทึกรายละเอียดข้อมูล
    data_info_file = os.path.join(PROJECT_PATH, 'data_info.txt')
    with open(data_info_file, 'w', encoding='utf-8') as f:
        f.write(f"จำนวนข้อมูลทั้งหมด: {len(X)} ภาพ\n\n")
        for letter in letters:
            count = y.count(letter)
            f.write(f"ตัวอักษร {letter}: {count} ภาพ\n")

    return np.array(X), np.array(y), file_paths


if __name__ == "__main__":
    collect_training_data()
    prepare_training_data()
