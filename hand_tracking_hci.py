"""
Implementasi Computer Vision pada Sistem Navigasi Mouse dan Pengaturan Audio 
Secara Real-time Berbasis Hand Tracking

Author: Senior Python CV Engineer
Description: Real-time HCI system using hand gestures to control mouse and system volume
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandTrackingController:
    """Main class for hand tracking-based HCI system"""
    
    def __init__(self, camera_index=0):
        """
        Initialize the Hand Tracking Controller
        
        Args:
            camera_index: Camera device index (0, 1) or IP address for DroidCam
        """
        # MediaPipe Hand Detection Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only 1 hand for stability
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video Capture Setup
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(3, 640)  # Width
        self.cap.set(4, 480)  # Height
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Frame reduction box (virtual tracking area)
        # This allows reaching screen corners without hand leaving camera view
        self.frame_reduction = 100  # pixels from each edge
        
        # Mouse smoothing variables
        self.prev_x, self.prev_y = 0, 0
        self.smooth_factor = 5  # Higher = smoother but slower response
        
        # Click debounce
        self.click_time = 0
        self.click_cooldown = 0.5  # seconds between clicks
        
        # Volume Control Setup (pycaw)
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.vol_range = self.volume.GetVolumeRange()  # (-65.25, 0.0, 0.03125)
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]
        
        # FPS calculation
        self.prev_time = 0
        
        # Hand landmark indices
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.THUMB_TIP = 4
        
    def calculate_distance(self, p1, p2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            p1: Point 1 (x, y)
            p2: Point 2 (x, y)
            
        Returns:
            float: Distance between points
        """
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def interpolate(self, value, in_min, in_max, out_min, out_max):
        """
        Map a value from one range to another (linear interpolation)
        
        Formula: out = out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
        
        Args:
            value: Input value
            in_min, in_max: Input range
            out_min, out_max: Output range
            
        Returns:
            float: Mapped value
        """
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
    
    def smooth_mouse_movement(self, x, y):
        """
        Apply exponential smoothing to reduce mouse jitter
        
        Args:
            x, y: Current mouse coordinates
            
        Returns:
            tuple: Smoothed (x, y) coordinates
        """
        # Exponential moving average
        smooth_x = self.prev_x + (x - self.prev_x) / self.smooth_factor
        smooth_y = self.prev_y + (y - self.prev_y) / self.smooth_factor
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)
    
    def control_mouse(self, landmarks, img_h, img_w):
        """
        Control mouse cursor using index finger tip
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            img_h, img_w: Image dimensions
        """
        # Get index finger tip coordinates
        index_tip = landmarks[self.INDEX_TIP]
        x = int(index_tip.x * img_w)
        y = int(index_tip.y * img_h)
        
        # Map coordinates from virtual box to screen resolution
        # Virtual box: (frame_reduction, frame_reduction) to (img_w - frame_reduction, img_h - frame_reduction)
        screen_x = self.interpolate(
            x, 
            self.frame_reduction, 
            img_w - self.frame_reduction,
            0, 
            self.screen_width
        )
        screen_y = self.interpolate(
            y, 
            self.frame_reduction, 
            img_h - self.frame_reduction,
            0, 
            self.screen_height
        )
        
        # Clamp values to screen bounds
        screen_x = np.clip(screen_x, 0, self.screen_width)
        screen_y = np.clip(screen_y, 0, self.screen_height)
        
        # Apply smoothing
        smooth_x, smooth_y = self.smooth_mouse_movement(screen_x, screen_y)
        
        # Move mouse
        pyautogui.moveTo(smooth_x, smooth_y)
        
        return x, y
    
    def detect_click(self, landmarks, img_h, img_w):
        """
        Detect click gesture (index finger + middle finger pinch)
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            img_h, img_w: Image dimensions
            
        Returns:
            bool: True if click detected
        """
        # Get fingertip coordinates
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        x1, y1 = int(index_tip.x * img_w), int(index_tip.y * img_h)
        x2, y2 = int(middle_tip.x * img_w), int(middle_tip.y * img_h)
        
        # Calculate distance
        distance = self.calculate_distance((x1, y1), (x2, y2))
        
        # Click threshold (fingers touching)
        click_threshold = 30
        
        # Check debounce cooldown
        current_time = time.time()
        if distance < click_threshold and (current_time - self.click_time) > self.click_cooldown:
            pyautogui.click()
            self.click_time = current_time
            return True
        
        return False
    
    def control_volume(self, landmarks, img_h, img_w, img):
        """
        Control system volume using thumb-index finger distance
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            img_h, img_w: Image dimensions
            img: Frame to draw on
            
        Returns:
            tuple: Volume bar coordinates and percentage
        """
        # Get thumb and index finger tip coordinates
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        
        x1, y1 = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
        x2, y2 = int(index_tip.x * img_w), int(index_tip.y * img_h)
        
        # Calculate distance
        distance = self.calculate_distance((x1, y1), (x2, y2))
        
        # Map distance to volume range
        # Distance range: approximately 20 (min) to 200 (max) pixels
        min_distance = 20
        max_distance = 200
        
        # Clamp distance
        distance = np.clip(distance, min_distance, max_distance)
        
        # Map to volume in decibels
        volume_db = self.interpolate(distance, min_distance, max_distance, self.min_vol, self.max_vol)
        self.volume.SetMasterVolumeLevel(volume_db, None)
        
        # Calculate volume percentage for display
        volume_percent = int(self.interpolate(distance, min_distance, max_distance, 0, 100))
        
        # Draw volume control visualization
        # Draw line between thumb and index
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        
        # Draw volume bar on left side
        bar_x, bar_y = 50, 100
        bar_width, bar_height = 50, 300
        
        # Background bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 3)
        
        # Filled bar (volume level)
        fill_height = int(self.interpolate(volume_percent, 0, 100, bar_height, 0))
        cv2.rectangle(img, (bar_x, bar_y + fill_height), (bar_x + bar_width, bar_y + bar_height), 
                     (0, 255, 0), cv2.FILLED)
        
        # Volume percentage text
        cv2.putText(img, f'{volume_percent}%', (bar_x - 10, bar_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return volume_percent
    
    def draw_virtual_box(self, img, img_h, img_w):
        """
        Draw the virtual tracking area on the frame
        
        Args:
            img: Frame to draw on
            img_h, img_w: Image dimensions
        """
        cv2.rectangle(img, 
                     (self.frame_reduction, self.frame_reduction),
                     (img_w - self.frame_reduction, img_h - self.frame_reduction),
                     (255, 0, 255), 2)
    
    def calculate_fps(self):
        """
        Calculate and return current FPS
        
        Returns:
            int: Frames per second
        """
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        return int(fps)
    
    def run(self):
        """Main loop for the hand tracking controller"""
        print("=" * 60)
        print("Hand Tracking HCI System Started")
        print("=" * 60)
        print("Controls:")
        print("  • Move Index Finger -> Control Mouse Cursor")
        print("  • Pinch Index + Middle Fingers -> Mouse Click")
        print("  • Adjust Thumb-Index Distance -> Control Volume")
        print("  • Press 'q' to Quit")
        print("=" * 60)
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture frame. Check camera connection.")
                break
            
            # Flip image horizontally for mirror effect
            img = cv2.flip(img, 1)
            img_h, img_w, _ = img.shape
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Draw virtual tracking box
            self.draw_virtual_box(img, img_h, img_w)
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmark list
                    landmarks = hand_landmarks.landmark
                    
                    # Highlight key fingertips
                    for tip_id in [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP]:
                        tip = landmarks[tip_id]
                        cx, cy = int(tip.x * img_w), int(tip.y * img_h)
                        cv2.circle(img, (cx, cy), 12, (0, 255, 255), cv2.FILLED)
                    
                    # Control mouse cursor
                    self.control_mouse(landmarks, img_h, img_w)
                    
                    # Detect click gesture
                    clicked = self.detect_click(landmarks, img_h, img_w)
                    if clicked:
                        cv2.putText(img, "CLICK!", (img_w // 2 - 50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Control volume
                    self.control_volume(landmarks, img_h, img_w, img)
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(img, f'FPS: {fps}', (img_w - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Display instructions
            cv2.putText(img, "Press 'q' to quit", (10, img_h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Hand Tracking HCI - Mouse & Volume Control", img)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nSystem terminated successfully.")


def main():
    """Main entry point"""
    # Camera Configuration
    # Use 0 for default webcam, 1 for second camera
    # For DroidCam IP: use "http://192.168.x.x:4747/video"
    CAMERA_INDEX = 0
    
    try:
        controller = HandTrackingController(camera_index=CAMERA_INDEX)
        controller.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure all required libraries are installed:")
        print("  pip install opencv-python mediapipe pyautogui pycaw comtypes numpy")


if __name__ == "__main__":
    main()
