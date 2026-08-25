# Sign Language Classification

โปรเจกต์สำหรับจำแนกตัวอักษรภาษามือภาษาอังกฤษ **A–I จำนวน 9 คลาส** จากภาพ โดยใช้ **Computer Vision** และ **Convolutional Neural Network (CNN)**

ระบบครอบคลุมตั้งแต่การเก็บข้อมูลภาพจากกล้อง การเตรียม Dataset การฝึกโมเดล ไปจนถึงการนำโมเดลมาใช้ทำนายตัวอักษรจากกล้องแบบ Real-time

---

## Objectives

- พัฒนาระบบสำหรับตรวจจับและจำแนกตัวอักษรภาษามือ A–I
- ฝึกการสร้างและเตรียม Dataset สำหรับใช้ในการฝึกโมเดล
- เรียนรู้การประยุกต์ใช้ Computer Vision และ Deep Learning กับข้อมูลภาพ
- ทดสอบการนำโมเดลที่ฝึกแล้วไปใช้งานกับกล้องแบบ Real-time

---

## Project Workflow

```text
Camera
   │
   ▼
Dataset Collection
   │
   ▼
Image Preprocessing
   │
   ▼
CNN Model Training
   │
   ▼
Trained Model
   │
   ▼
Real-time Prediction
