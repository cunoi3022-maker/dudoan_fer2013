import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
import joblib

# === Load mô hình đã lưu ===
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
rf = joblib.load("rf_model.joblib")

# === Hàm tiền xử lý ảnh khuôn mặt ===
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_img = clahe.apply(face_img)
    return face_img

# === Hàm xoay ảnh ===
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# === Hàm phát hiện khuôn mặt với nhiều góc xoay ===
def detect_faces_with_rotation(gray_image, angles=[0, 90, 180, 270]):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for angle in angles:
        rotated = rotate_image(gray_image, angle) if angle != 0 else gray_image
        faces = face_cascade.detectMultiScale(rotated, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            return faces, angle, rotated

    return [], 0, gray_image  # không phát hiện khuôn mặt

# === Tiêu đề ứng dụng ===
st.title("📸 Ứng dụng nhận diện cảm xúc khuôn mặt (Fer - 2013)")
st.write("Tải ảnh khuôn mặt lên để mô hình dự đoán cảm xúc (tự xử lý ảnh bị xoay)")

uploaded_file = st.file_uploader("📷 Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh gốc
    image = Image.open(uploaded_file).convert("RGB")
    img_np_original = np.array(image)
    gray = cv2.cvtColor(img_np_original, cv2.COLOR_RGB2GRAY)

    # Thử phát hiện khuôn mặt với nhiều góc xoay
    faces, used_angle, rotated_gray = detect_faces_with_rotation(gray)

    # Xoay ảnh RGB tương ứng để hiển thị kết quả đúng chiều
    rotated_rgb = rotate_image(img_np_original, used_angle) if used_angle != 0 else img_np_original

    st.image(rotated_rgb, caption=f"Phát hiện {len(faces)} khuôn mặt (góc xoay: {used_angle}°)", use_column_width=True)

    # Dự đoán cảm xúc cho từng khuôn mặt
    for (x, y, w, h) in faces:
        face = rotated_gray[y:y + h, x:x + w]
        face_processed = preprocess_face(face)

        # === HOG đặc trưng ===
        features = hog(
            face_processed,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        ).reshape(1, -1)

        # === Chuẩn hóa + PCA ===
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # === Dự đoán cảm xúc ===
        prediction = rf.predict(features_pca)[0]

        # === Vẽ kết quả lên ảnh đã xoay ===
        cv2.rectangle(rotated_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            rotated_rgb,
            prediction.capitalize(),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Hiển thị ảnh kết quả cuối
    st.image(rotated_rgb, caption="--> Kết quả nhận diện cảm xúc", use_column_width=True)
