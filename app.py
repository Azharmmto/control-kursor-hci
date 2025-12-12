"""
Hand Tracking Mouse & Volume Control (FINAL STABLE VERSION)
Fitur: 
1. Mouse: Jari Telunjuk (Tangan Terbuka)
2. Klik: Rapatkan Telunjuk & Tengah
3. Volume: Hanya aktif jika Jari Kelingking DITEKUK (Safety Lock)

Author: Senior Python CV Engineer
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Matikan delay safety agar responsif
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

class HandTrackingController:
    def __init__(self, camera_index=2):
        # 1. Setup Kamera
        print(f"Mencoba koneksi kamera index {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        
        # Auto-switch logic
        if not self.cap.isOpened() or not self.cap.read()[0]:
            print(f"Index {camera_index} gagal. Mencoba mencari kamera lain...")
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

        # 2. MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen Setup
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_reduction = 100 
        
        # --- TUNING VARIABLES ---
        self.smooth_factor = 5      # 5 = Cukup responsif, tidak terlalu lambat
        self.dead_zone = 3          # Mengurangi getaran kursor
        
        self.click_threshold = 30   # Jarak klik (Telunjuk + Tengah)
        self.vol_up_thresh = 110    # Harus merentang lebar untuk Volume Up
        self.vol_down_thresh = 30   # Harus mencubit rapat untuk Volume Down
        
        # State Variables
        self.prev_x, self.prev_y = 0, 0
        self.click_time = 0
        self.vol_time = 0     
        self.click_cooldown = 0.5
        self.vol_cooldown = 0.15    # Sedikit diperlambat agar tidak terlalu sensitif
        
        # Finger Indices
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        self.PINKY_MCP = 17 # Pangkal jari kelingking
        
        self.prev_time = 0

    def calculate_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    def interpolate(self, value, in_min, in_max, out_min, out_max):
        return np.interp(value, [in_min, in_max], [out_min, out_max])
    
    def move_mouse_smooth(self, x, y):
        # Mapping koordinat
        target_x = self.interpolate(x, self.frame_reduction, 640 - self.frame_reduction, 0, self.screen_width)
        target_y = self.interpolate(y, self.frame_reduction, 480 - self.frame_reduction, 0, self.screen_height)
        
        # Clip
        target_x = np.clip(target_x, 0, self.screen_width - 1)
        target_y = np.clip(target_y, 0, self.screen_height - 1)
        
        # Smoothing
        curr_x = self.prev_x + (target_x - self.prev_x) / self.smooth_factor
        curr_y = self.prev_y + (target_y - self.prev_y) / self.smooth_factor
        
        # Deadzone (Anti-Jitter)
        dist_moved = math.hypot(curr_x - self.prev_x, curr_y - self.prev_y)
        if dist_moved < self.dead_zone:
            return 
        
        self.prev_x, self.prev_y = curr_x, curr_y
        pyautogui.moveTo(int(curr_x), int(curr_y))

    def is_pinky_folded(self, landmarks):
        """
        Cek apakah jari kelingking ditekuk.
        Jika Ujung Kelingking (20) posisinya lebih bawah dari Pangkal Kelingking (17),
        berarti jari ditekuk (Koordinat Y makin ke bawah makin besar).
        """
        pinky_tip = landmarks[self.PINKY_TIP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        
        # Logika: Jika Tip.Y > MCP.Y berarti ujung jari ada di bawah pangkal jari (Menekuk)
        return pinky_tip.y > pinky_mcp.y

    def process_hand(self, landmarks, img_h, img_w, img):
        # Ambil koordinat penting
        index_tip = landmarks[self.INDEX_TIP]
        thumb_tip = landmarks[self.THUMB_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        ix, iy = int(index_tip.x * img_w), int(index_tip.y * img_h)
        tx, ty = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
        mx, my = int(middle_tip.x * img_w), int(middle_tip.y * img_h)

        # --- 1. MODE CHECKER ---
        # Cek apakah jari kelingking ditekuk?
        volume_mode_active = self.is_pinky_folded(landmarks)

        # --- 2. MOUSE MOVEMENT (Selalu Aktif) ---
        # Visualisasi jari telunjuk
        cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
        self.move_mouse_smooth(ix, iy)
        
        # --- 3. CLICK DETECTION (Telunjuk + Tengah Rapat) ---
        click_dist = self.calculate_distance((ix, iy), (mx, my))
        
        if click_dist < self.click_threshold:
            # Visualisasi Klik
            cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "CLICK", (ix, iy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if (time.time() - self.click_time) > self.click_cooldown:
                pyautogui.click()
                self.click_time = time.time()
                
            # SAFETY: Jika sedang klik, jangan jalankan volume
            return 

        # --- 4. VOLUME CONTROL (Hanya jika Kelingking Ditekuk) ---
        if volume_mode_active:
            # Tampilkan Indikator Mode Volume Aktif
            cv2.putText(img, "MODE: VOLUME (Active)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hitung Jarak Jempol - Telunjuk
            vol_dist = self.calculate_distance((tx, ty), (ix, iy))
            
            # Gambar Garis Penghubung
            cx, cy = (tx + ix) // 2, (ty + iy) // 2
            cv2.line(img, (tx, ty), (ix, iy), (0, 255, 0), 2)
            cv2.putText(img, f"Dist: {int(vol_dist)}", (cx, cy+30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

            curr_time = time.time()
            if (curr_time - self.vol_time) > self.vol_cooldown:
                if vol_dist < self.vol_down_thresh: # Cubit
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                    pyautogui.press('volumedown')
                    self.vol_time = curr_time
                    
                elif vol_dist > self.vol_up_thresh: # Rentang
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    pyautogui.press('volumeup')
                    self.vol_time = curr_time
        else:
            # Jika mode Mouse biasa (Kelingking Lurus)
            cv2.putText(img, "MODE: MOUSE (Vol Locked)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run(self):
        print("\n=== SYSTEM STARTED ===")
        print("ATURAN BARU:")
        print("1. Mouse & Klik: Gunakan tangan terbuka biasa.")
        print("2. Volume: TEKUK Jari Kelingking & Manis (Pose Pistol/L).")
        print("   -> Volume terkunci jika kelingking lurus.")
        print("Tekan 'q' untuk keluar.")
        
        while True:
            success, img = self.cap.read()
            if not success: break
            
            img = cv2.flip(img, 1)
            img_h, img_w, _ = img.shape
            
            cv2.rectangle(img, (self.frame_reduction, self.frame_reduction), 
                         (img_w - self.frame_reduction, img_h - self.frame_reduction), (255, 0, 255), 1)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.process_hand(hand_landmarks.landmark, img_h, img_w, img)
            
            # FPS
            c_time = time.time()
            fps = 1 / (c_time - self.prev_time) if c_time > self.prev_time else 0
            self.prev_time = c_time
            cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Hand Tracking Control", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    HandTrackingController(camera_index=2).run()