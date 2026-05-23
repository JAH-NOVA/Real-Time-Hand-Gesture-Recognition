import cv2
import mediapipe as mp
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Function to find distance between 2 points
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gestures = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            hand_label = handedness.classification[0].label

            # Palm size for scaling
            palm = distance(lm[0], lm[9])
            if palm == 0:
                palm = 1

            # Finger states
            index = lm[8].y < lm[6].y
            middle = lm[12].y < lm[10].y
            ring = lm[16].y < lm[14].y
            pinky = lm[20].y < lm[18].y

            # Thumb state (for side-open poses like HELLO / LOVE)
            if hand_label == "Right":
                thumb = lm[4].x < lm[2].x
            else:
                thumb = lm[4].x > lm[2].x

            # Distance for OK gesture
            dist = distance(lm[4], lm[8])

            # Thumb extension for YES / BAD / FIST
            thumb_length = distance(lm[4], lm[2]) / palm

            all_folded = (not index and not middle and not ring and not pinky)

            # Gesture detection
            if dist < 0.05 and middle and ring and pinky:
                gestures.append(f"{hand_label}: OK")
                 

            elif thumb and index and not middle and not ring and pinky:
                gestures.append(f"{hand_label}: LOVE")   

            elif all_folded and thumb_length > 0.55 and lm[4].y < lm[3].y < lm[2].y:
                gestures.append(f"{hand_label}: YES")   

            elif all_folded and thumb_length > 0.55 and lm[4].y > lm[3].y > lm[2].y:
                gestures.append(f"{hand_label}: BAD")     

            elif all_folded:
                gestures.append(f"{hand_label}: FIST")      

            elif thumb and index and middle and ring and pinky:
                gestures.append(f"{hand_label}: HELLO")     

            elif not thumb and index and middle and ring and pinky:
                gestures.append(f"{hand_label}: FOUR")   

            elif not thumb and index and middle and ring and not pinky:
                gestures.append(f"{hand_label}: THREE")    

            elif not thumb and index and middle and not ring and not pinky:
                gestures.append(f"{hand_label}: TWO")     

            elif not thumb and index and not middle and not ring and not pinky:
                gestures.append(f"{hand_label}: NO")   

            else:
                gestures.append(f"{hand_label}: UNKNOWN") 

            # Show left/right hand
             
            h, w, c = frame.shape

            x = int(lm[0].x * w)
            y = int(lm[0].y * h)

            cv2.putText(frame,
                        hand_label,
                        (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2)

    # Show detected gesture
     
         
    y = 80
    for g in gestures:
        cv2.putText(frame, g, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3)
        y += 40

    cv2.imshow("Sign Language Translator Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
