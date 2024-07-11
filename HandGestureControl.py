import cv2
import mediapipe as mp
import time
import pyautogui

# Initialize hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions (replace with your actual screen dimensions)
screen_width, screen_height = 1366, 768

# Define finger tip IDs
thumb_tip_id = 4
index_tip_id = 8
middle_tip_id = 12

# Threshold for click detection based on fingertip distance
click_threshold = 20
middle_finger_tip_y_threshold = 150

# Click state variables (single, double, hold)
single_click_time = 0
double_click_threshold = 0.3  # Seconds between clicks for double click
is_holding_click = False

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    success, img = cap.read()

    # Convert image to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark data for thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[thumb_tip_id]
            index_tip = hand_landmarks.landmark[index_tip_id]

            # Calculate fingertip positions in image coordinates
            thumb_cx, thumb_cy = int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0])
            index_cx, index_cy = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])

            # Calculate fingertip distance
            finger_distance = ((thumb_cx - index_cx) ** 2 + (thumb_cy - index_cy) ** 2) ** 0.5

            # Update cursor position
            cursor_x = index_cx
            cursor_y = index_cy

            # Draw circle on fingertips (optional)
            cv2.circle(img, (thumb_cx, thumb_cy), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_cx, index_cy), 10, (255, 0, 255), cv2.FILLED)

            # Map cursor position to screen coordinates
            mapped_cursor_x = int(cursor_x * (screen_width / img.shape[1]))
            mapped_cursor_y = int(cursor_y * (screen_height / img.shape[0]))

            # Move the cursor to the mapped coordinates
            pyautogui.moveTo(mapped_cursor_x, mapped_cursor_y)

            # Click detection logic
            if finger_distance <= click_threshold and not is_holding_click:
                current_time = time.time()

                # Single click detection
                if abs(current_time - single_click_time) > double_click_threshold:
                    single_click_time = current_time
                    print("Single Left Click")

                # Double click detection
                else:
                    double_click_time = current_time
                    print("Double Left Click")
                    # Simulate left click here (e.g., using external libraries)

            # Holding click logic
            if finger_distance <= click_threshold:
                is_holding_click = True
            else:
                is_holding_click = False

            # Draw cursor indicator (optional)
            cv2.circle(img, (cursor_x, cursor_y), 5, (0, 255, 0), cv2.FILLED)

            # Map cursor position to screen coordinates (adjust based on your setup)
            mapped_cursor_x = int(cursor_x * (screen_width / img.shape[1]))
            mapped_cursor_y = int(cursor_y * (screen_height / img.shape[0]))

    # Display the resulting frame
    cv2.imshow('Hand Control', img)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
