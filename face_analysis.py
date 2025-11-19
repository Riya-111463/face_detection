import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access camera.")
        break

    try:
        # Analyze the frame using DeepFace
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        # Extract information
        age = result[0]['age']
        gender = result[0]['gender']
        emotion = result[0]['dominant_emotion']

        # Display the data on screen
        text = f"{gender}, Age: {age}, Emotion: {emotion}"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    except Exception as e:
        print("No face detected in this frame.")

    cv2.imshow("Real-Time Face, Age, Gender & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
