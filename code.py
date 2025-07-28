import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Khởi tạo Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Đọc hai hình ảnh
img1 = cv2.imread('khong_lech.jpg')
img2 = cv2.imread('lech.jpg')

# Kiểm tra xem hình ảnh có được đọc thành công không
if img1 is None:
    print("Lỗi: Không thể đọc khong_lech.jpg. Kiểm tra tên file và đường dẫn.")
    exit()
if img2 is None:
    print("Lỗi: Không thể đọc lech.jpg. Kiểm tra tên file và đường dẫn.")
    exit()

print(f"Kích thước img1: {img1.shape}, img2: {img2.shape}")

# Thay đổi kích thước hình ảnh để hiển thị nhỏ hơn
scale_percent = 50
width1 = int(img1.shape[1] * scale_percent / 100)
height1 = int(img1.shape[0] * scale_percent / 100)
width2 = int(img2.shape[1] * scale_percent / 100)
height2 = int(img2.shape[0] * scale_percent / 100)
img1_resized = cv2.resize(img1, (width1, height1), interpolation=cv2.INTER_AREA)
img2_resized = cv2.resize(img2, (width2, height2), interpolation=cv2.INTER_AREA)

# Chuyển sang RGB
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Phát hiện landmarks
results1 = face_mesh.process(img1_rgb)
results2 = face_mesh.process(img2_rgb)

print(f"Landmarks detected for img1: {results1.multi_face_landmarks is not None}")
print(f"Landmarks detected for img2: {results2.multi_face_landmarks is not None}")

def calculate_symmetry(landmarks, img_shape):
    symmetry_pairs = [(33, 263), (61, 291), (4, 298)]
    distances = []
    
    for left_idx, right_idx in symmetry_pairs:
        left_point = landmarks.landmark[left_idx]
        right_point = landmarks.landmark[right_idx]
        left_x, left_y = left_point.x * img_shape[1], left_point.y * img_shape[0]
        right_x, right_y = right_point.x * img_shape[1], right_point.y * img_shape[0]
        distance = np.sqrt((left_x - right_x)**2 + (left_y - right_y)**2)
        distances.append(distance)
    
    return np.mean(distances) / img_shape[1] * 100

# Kiểm tra và tính toán độ lệch
if results1.multi_face_landmarks and results2.multi_face_landmarks:
    landmarks1 = results1.multi_face_landmarks[0]
    landmarks2 = results2.multi_face_landmarks[0]
    
    asymmetry1 = calculate_symmetry(landmarks1, img1.shape)
    asymmetry2 = calculate_symmetry(landmarks2, img2.shape)
    
    print(f"Asymmetry 1: {asymmetry1:.2f}%, Asymmetry 2: {asymmetry2:.2f}%")
    
    results = []
    for i, asymmetry in enumerate([asymmetry1, asymmetry2], 1):
        if asymmetry < 20:
            level = "Lệch nhẹ"
        elif asymmetry < 40:
            level = "Lệch trung bình"
        else:
            level = "Lệch nhiều"
        print(f"Hình {i}: {level}")
        results.append({"Image": f"Image {i}", "Asymmetry (%)": asymmetry, "Level": level})
    
    mp_drawing.draw_landmarks(img1_resized, landmarks1, mp_face_mesh.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(img2_resized, landmarks2, mp_face_mesh.FACEMESH_TESSELATION)
    
    cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Image 1', img1_resized)
    cv2.imshow('Image 2', img2_resized)
    
    plt.figure(figsize=(6, 4))
    plt.bar(["Image 1", "Image 2"], [asymmetry1, asymmetry2], color=['#FF9999', '#66B2FF'])
    plt.title('Degree of Facial Asymmetry')
    plt.xlabel('Image')
    plt.ylabel('Asymmetry (%)')
    plt.ylim(0, max(asymmetry1, asymmetry2) * 1.2)
    for i, v in enumerate([asymmetry1, asymmetry2]):
        plt.text(i, v + 1, f'{v:.2f}%', ha='center')
    plt.show()
    
    df = pd.DataFrame(results)
    df.to_csv('face_asymmetry_results.csv', index=False)
    print("Kết quả đã được lưu vào 'face_asymmetry_results.csv'")
else:
    print("Lỗi: Không phát hiện được gương mặt trong một hoặc cả hai hình ảnh.")
    print("Đề xuất: Đảm bảo hình ảnh chứa gương mặt chính diện, rõ ràng, và ánh sáng tốt.")

face_mesh.close()