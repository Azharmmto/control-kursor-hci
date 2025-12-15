import os
import warnings

# --- Sembunyikan Warning ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Konfigurasi PyAutoGUI
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

class HandTrackingController:
    def __init__(self, camera_index=2):
        # --- SETUP KAMERA ---
        print(f"Mencoba membuka kamera index {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        
        # Fallback Logic
        if not self.cap.isOpened() or not self.cap.read()[0]:
            print(f"❌ Index {camera_index} gagal. Mencoba mencari kamera lain...")
            for i in range(3):
                if i == camera_index: continue
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened() and temp_cap.read()[0]:
                    print(f"✅ Kamera ditemukan pada Index: {i}")
                    self.cap = temp_cap
                    break
                temp_cap.release()

        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # --- MEDIAPIPE ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # --- SCREEN & VARIABLES ---
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_reduction = 100 
        self.smooth_factor = 5
        self.dead_zone = 3
        self.prev_x, self.prev_y = 0, 0
        
        # Thresholds (Pixel)
        self.click_dist_thresh = 40  # Jarak sentuh untuk Klik
        self.vol_up_thresh = 80      # Jarak rentang Volume Up
        self.vol_down_thresh = 30    # Jarak cubit Volume Down
        self.fingers_touch_thresh = 45 # Jarak Telunjuk & Tengah (Scroll Mode)
        
        # Timers
        self.last_action_time = 0
        self.click_cooldown = 0.5
        self.vol_cooldown = 0.1
        self.prev_time = 0
        
        # Scroll Logic Variables
        self.scroll_anchor_y = 0     # Titik awal scroll
        self.scroll_speed_factor = 3 # Pengali kecepatan scroll (makin besar makin ngebut)
        self.scroll_dead_zone = 20   # Area tengah agar scroll tidak sensitif getaran
        
        # Indices Landmark
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.THUMB_IP = 3   # Sendi Jempol
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        # MCP (Pangkal Jari)
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17

    def calculate_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def interpolate(self, value, in_min, in_max, out_min, out_max):
        return np.interp(value, [in_min, in_max], [out_min, out_max])
    
    def is_finger_folded(self, landmarks, tip_idx, mcp_idx):
        # Jari dianggap tertutup jika Ujungnya lebih rendah (Y lebih besar) dari Pangkalnya
        return landmarks[tip_idx].y > landmarks[mcp_idx].y

    def is_thumb_folded(self, landmarks):
        # Cek jempol tertutup (Ujung jempol dekat ke pangkal jari telunjuk/tengah)
        thumb_tip = landmarks[self.THUMB_TIP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        # Logika simpel: X jempol melewati X telunjuk (tergantung tangan kiri/kanan)
        # Atau cek jarak ke jari kelingking
        return self.calculate_distance((thumb_tip.x, thumb_tip.y), (pinky_mcp.x, pinky_mcp.y)) < 0.2

    def get_gesture_mode(self, landmarks, img_w, img_h):
        """Menentukan Mode Operasi"""
        # Cek status jari (Buka/Tutup)
        thumb_folded = self.is_thumb_folded(landmarks) # Opsional, kadang jempol susah dideteksi tutup
        index_folded = self.is_finger_folded(landmarks, self.INDEX_TIP, self.INDEX_MCP)
        middle_folded = self.is_finger_folded(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP)
        ring_folded = self.is_finger_folded(landmarks, self.RING_TIP, self.RING_MCP)
        pinky_folded = self.is_finger_folded(landmarks, self.PINKY_TIP, self.PINKY_MCP)

        # Koordinat Pixel untuk cek jarak
        ix, iy = int(landmarks[self.INDEX_TIP].x * img_w), int(landmarks[self.INDEX_TIP].y * img_h)
        mx, my = int(landmarks[self.MIDDLE_TIP].x * img_w), int(landmarks[self.MIDDLE_TIP].y * img_h)
        dist_index_middle = self.calculate_distance((ix, iy), (mx, my))

        # 1. SCROLL MODE:
        # Syarat: Telunjuk & Tengah BUKA & RAPAT, Sisanya (Manis, Kelingking) TUTUP.
        # Jempol sebaiknya tutup atau netral.
        if (not index_folded) and (not middle_folded) and ring_folded and pinky_folded:
            if dist_index_middle < self.fingers_touch_thresh: # Dua jari rapat
                return "SCROLL"
        
        # 2. VOLUME MODE:
        # Syarat: Tengah, Manis, Kelingking TUTUP. Telunjuk & Jempol BUKA.
        if (not index_folded) and middle_folded and ring_folded and pinky_folded:
            return "VOLUME"
            
        # 3. MOUSE MODE:
        # Syarat: Kelingking BUKA (Syarat utama user)
        if not pinky_folded:
            return "MOUSE"
            
        return "UNKNOWN"

    def move_mouse(self, x, y):
        # Mapping koordinat
        tx = self.interpolate(x, self.frame_reduction, 640 - self.frame_reduction, 0, self.screen_width)
        ty = self.interpolate(y, self.frame_reduction, 480 - self.frame_reduction, 0, self.screen_height)
        
        tx = np.clip(tx, 0, self.screen_width - 1)
        ty = np.clip(ty, 0, self.screen_height - 1)
        
        cx = self.prev_x + (tx - self.prev_x) / self.smooth_factor
        cy = self.prev_y + (ty - self.prev_y) / self.smooth_factor
        
        if math.hypot(cx - self.prev_x, cy - self.prev_y) < self.dead_zone:
            return

        self.prev_x, self.prev_y = cx, cy
        pyautogui.moveTo(int(cx), int(cy))

    def process_hand(self, landmarks, img_h, img_w, img):
        # Koordinat
        wrist = landmarks[self.WRIST]
        thumb = landmarks[self.THUMB_TIP]
        index = landmarks[self.INDEX_TIP]
        middle = landmarks[self.MIDDLE_TIP]
        
        wx, wy = int(wrist.x * img_w), int(wrist.y * img_h)
        ix, iy = int(index.x * img_w), int(index.y * img_h)
        tx, ty = int(thumb.x * img_w), int(thumb.y * img_h)
        mx, my = int(middle.x * img_w), int(middle.y * img_h)

        mode = self.get_gesture_mode(landmarks, img_w, img_h)
        curr_time = time.time()

        # ================= LOGIKA MODE =================
        
        # --- 1. SCROLL MODE (Dua Jari Rapat) ---
        if mode == "SCROLL":
            # Tampilkan Indikator
            cv2.putText(img, "MODE: SCROLL", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # Visualisasi Dua Jari
            cx, cy = (ix + mx) // 2, (iy + my) // 2
            cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
            cv2.line(img, (ix, iy), (mx, my), (255, 255, 0), 2)
            
            # Set Anchor (Titik Tengah) jika baru masuk mode
            if self.scroll_anchor_y == 0:
                self.scroll_anchor_y = cy

            # Hitung jarak jari dari Anchor
            delta_y = cy - self.scroll_anchor_y
            
            # Visualisasi Garis Anchor ke Jari
            cv2.line(img, (cx, self.scroll_anchor_y), (cx, cy), (100, 100, 100), 2)
            cv2.circle(img, (cx, self.scroll_anchor_y), 5, (0, 0, 0), cv2.FILLED) # Titik Anchor

            # Logika Smooth Scroll (Velocity Based)
            if abs(delta_y) > self.scroll_dead_zone:
                # Arah scroll: delta_y Positif (Tangan Turun) -> Scroll Bawah
                # PyAutoGUI: scroll negatif = bawah, positif = atas
                
                scroll_amount = int(delta_y / 2) # Sesuaikan pembagi untuk sensitivitas
                
                # Invert logic: Tangan ke Atas (y kecil, delta negatif) -> Scroll Up (+)
                # Tangan ke Bawah (y besar, delta positif) -> Scroll Down (-)
                pyautogui.scroll(-scroll_amount)
                
                direction = "DOWN" if delta_y > 0 else "UP"
                cv2.putText(img, f"SCROLL {direction}", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # JANGAN reset anchor_y agar scroll terasa seperti joystick (continuous)
            # User harus mengembalikan tangan ke posisi awal (anchor) untuk stop scroll

        # --- 2. VOLUME MODE (Pistol Pose) ---
        elif mode == "VOLUME":
            self.scroll_anchor_y = 0 # Reset scroll
            cv2.putText(img, "MODE: VOLUME", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            vol_dist = self.calculate_distance((tx, ty), (ix, iy))
            cx, cy = (tx + ix) // 2, (ty + iy) // 2
            
            cv2.line(img, (tx, ty), (ix, iy), (0, 255, 0), 2)
            cv2.putText(img, f"Dist: {int(vol_dist)}", (cx, cy+30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

            if (curr_time - self.last_action_time) > self.vol_cooldown:
                if vol_dist < self.vol_down_thresh: # Cubit -> Vol Down
                    pyautogui.press('volumedown')
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                    self.last_action_time = curr_time
                elif vol_dist > self.vol_up_thresh: # Rentang -> Vol Up
                    pyautogui.press('volumeup')
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                    self.last_action_time = curr_time

        # --- 3. MOUSE MODE (Kelingking Terbuka) ---
        elif mode == "MOUSE":
            self.scroll_anchor_y = 0 # Reset scroll
            cv2.putText(img, "MODE: MOUSE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Gerak Mouse (Telunjuk)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            self.move_mouse(ix, iy)
            
            # A. KLIK KIRI (Jempol + Telunjuk)
            left_click_dist = self.calculate_distance((tx, ty), (ix, iy))
            
            # B. KLIK KANAN (Jempol + Tengah)
            right_click_dist = self.calculate_distance((tx, ty), (mx, my))
            
            # Eksekusi Klik Kiri
            if left_click_dist < self.click_dist_thresh:
                cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "L-CLICK", (ix, iy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if (curr_time - self.last_action_time) > self.click_cooldown:
                    pyautogui.click()
                    self.last_action_time = curr_time
                return 

            # Eksekusi Klik Kanan
            if right_click_dist < self.click_dist_thresh:
                cv2.circle(img, (mx, my), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "R-CLICK", (mx, my-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if (curr_time - self.last_action_time) > self.click_cooldown:
                    pyautogui.rightClick()
                    self.last_action_time = curr_time
        
        else:
            # Tidak ada mode yang cocok (Reset Anchor)
            self.scroll_anchor_y = 0

    def run(self):
        print("\n=== SYSTEM STARTED ===")
        print("Camera Index:", CAMERA_TARGET)
        print("1. Mouse (Kelingking Buka): Klik Kiri(Jempol+Telunjuk), Kanan(Jempol+Tengah)")
        print("2. Scroll (Telunjuk & Tengah Rapat + Sisa Tutup): Gerak Atas/Bawah")
        print("3. Volume (Pistol Pose): Rentang Jempol-Telunjuk")
        print("Tekan 'q' untuk keluar.")
        
        while True:
            success, img = self.cap.read()
            if not success: break
            
            img = cv2.flip(img, 1)
            img_h, img_w, _ = img.shape
            
            # Area Mouse
            cv2.rectangle(img, (self.frame_reduction, self.frame_reduction), 
                         (img_w - self.frame_reduction, img_h - self.frame_reduction), (255, 0, 255), 1)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.process_hand(hand_landmarks.landmark, img_h, img_w, img)
            else:
                self.scroll_anchor_y = 0 # Reset jika tangan hilang
            
            # FPS Calculation
            c_time = time.time()
            if (c_time - self.prev_time) > 0:
                fps = 1 / (c_time - self.prev_time)
            else:
                fps = 0
            self.prev_time = c_time
            
            cv2.putText(img, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Hand Control", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CAMERA_TARGET = 2  
    HandTrackingController(camera_index=CAMERA_TARGET).run()