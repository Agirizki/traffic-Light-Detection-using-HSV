import cv2
import numpy as np
import time

# === SETUP VIDEO INPUT ===
video_path = "Traffic Light Animation.mp4"  # Ganti dengan path video Anda
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Tidak dapat membuka video.")
    exit()

# === RENTANG WARNA HSV UNTUK DETEKSI LAMPU ===
# Rentang untuk warna merah
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])
# Rentang untuk warna kuning
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
# Rentang untuk warna hijau
green_lower = np.array([40, 100, 100])
green_upper = np.array([70, 255, 255])

# === FUNGSI UNTUK DETEKSI WARNA ===
def detect_color(hsv_frame):
    # Mask untuk merah
    red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2

    # Mask untuk kuning
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

    # Mask untuk hijau
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    return red_mask, yellow_mask, green_mask

# === FUNGSI UNTUK MENGGAMBAR BOUNDING BOX DAN LABEL ===
def draw_bounding_box(frame, mask, color_label, box_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter area kecil (noise)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

# === FUNGSI UNTUK MENINGKATKAN KONDISI LOW-LIGHT ===
def enhance_brightness(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_frame = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)

# === LOG DETEKSI ===
log_file = open("traffic_light_log.txt", "w")

def write_log(color, timestamp):
    log_file.write(f"{timestamp:.2f}: {color}\n")

# === MAIN PROGRAM ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tingkatkan kecerahan untuk kondisi low-light
    enhanced_frame = enhance_brightness(frame)

    # Konversi ke HSV
    hsv_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)

    # Deteksi warna
    red_mask, yellow_mask, green_mask = detect_color(hsv_frame)

    # Catat warna yang terdeteksi ke log
    current_time = time.time()
    if cv2.countNonZero(red_mask) > 0:
        write_log("Red", current_time)
    if cv2.countNonZero(yellow_mask) > 0:
        write_log("Yellow", current_time)
    if cv2.countNonZero(green_mask) > 0:
        write_log("Green", current_time)

    # Gambar bounding box di sekitar lampu
    draw_bounding_box(frame, red_mask, "Red", (0, 0, 255))
    draw_bounding_box(frame, yellow_mask, "Yellow", (0, 255, 255))
    draw_bounding_box(frame, green_mask, "Green", (0, 255, 0))

    # Tampilkan hasil
    cv2.imshow("Traffic Light Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
log_file.close()
cap.release()
cv2.destroyAllWindows()
print("Proses selesai. Log disimpan di 'traffic_light_log.txt'.")
