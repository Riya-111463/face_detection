import cv2
from deepface import DeepFace
import numpy as np
import time

# Adjust these thresholds as you wish
YOUNG_MAX = 29      # age <= 29 => 'Young'
ADULT_MAX = 59      # 30-59 => 'Adult', 60+ => 'Old'

# Map DeepFace emotion to simpler mood labels (you can alter)
MOOD_MAP = {
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'surprise': 'Surprised',
    'neutral': 'Neutral',
    'disgust': 'Disgusted',
    'fear': 'Afraid',
    'contempt': 'Contempt'
}

def age_category(age):
    try:
        a = int(round(age))
    except Exception:
        return "Unknown"
    if a <= YOUNG_MAX:
        return "Young"
    if a <= ADULT_MAX:
        return "Adult"
    return "Old"

def map_mood(emotion_str):
    if not emotion_str:
        return "Unknown"
    return MOOD_MAP.get(emotion_str.lower(), emotion_str.title())

def process_frame(frame):
    """
    Analyze a frame with DeepFace.
    Returns a list of dictionaries for each detected face with keys:
      'age', 'gender', 'dominant_emotion', 'mood', 'age_category', 'region' (bbox)
    """
    try:
        # Analyze may return a list (for multiple faces) or dict (single face)
        results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        # Normalize to a list
        if isinstance(results, dict):
            results = [results]

        faces_info = []
        for res in results:
            # DeepFace returns 'region' for face bbox when available
            region = res.get('region', None)
            age = res.get('age', None)
            gender = res.get('gender', None)
            dominant_emotion = res.get('dominant_emotion', None)

            face_info = {
                'age': age,
                'gender': gender,
                'dominant_emotion': dominant_emotion,
                'mood': map_mood(dominant_emotion),
                'age_category': age_category(age),
                'region': region
            }
            faces_info.append(face_info)

        return faces_info
    except Exception as e:
        # Log error and return empty list (so UI keeps running)
        print("Analysis error:", str(e))
        return []

def draw_info_on_frame(frame, faces_info):
    # For better placement when no 'region' provided, we'll place text at top-left and then shift
    h, w = frame.shape[:2]
    base_y = 30
    line_height = 26
    i = 0
    for face in faces_info:
        text = f"{face['mood']} | {face['gender']} | Age: {face['age']} ({face['age_category']})"
        y = base_y + i * line_height
        # Draw a semi-opaque rectangle behind text for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, y - 20), (15 + tw, y + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If bounding box is available, draw it and label near the box
        region = face.get('region')
        if region and all(k in region for k in ('x','y','w','h')):
            x, y0, w0, h0 = int(region['x']), int(region['y']), int(region['w']), int(region['h'])
            cv2.rectangle(frame, (x, y0), (x + w0, y0 + h0), (255, 0, 0), 2)
            label = f"{face['mood']}, {face['gender']}, {face['age_category']}"
            cv2.putText(frame, label, (x, max(10, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        i += 1

    # If no faces detected, optionally show a hint
    if len(faces_info) == 0:
        cv2.putText(frame, "No face detected / analysis failed", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    print("Press 'q' to quit. Note: Model predicts gender (Male/Female) but cannot determine transgender identity.")
    last_analysis_time = 0
    analysis_interval = 0.8  # seconds between heavy analyses (reduce load)

    faces_info_cache = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera frame not available.")
            break

        # Do heavy DeepFace analyze only at intervals to improve FPS
        now = time.time()
        if now - last_analysis_time >= analysis_interval:
            faces_info_cache = process_frame(frame)
            last_analysis_time = now

        out_frame = draw_info_on_frame(frame.copy(), faces_info_cache)

        cv2.imshow("Face Mood / Gender / Age Detection", out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
