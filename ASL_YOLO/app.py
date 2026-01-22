import cv2
from ultralytics import YOLO
import pyttsx3
import time


model = YOLO("best.pt")  


engine = pyttsx3.init()
engine.setProperty('rate', 150)


cap = cv2.VideoCapture(0)

letters = []  
last_letter = None
last_time = 0
delay = 1.0    

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)


    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        cls = int(box.cls[0])
        letter = model.names[cls]

        current_time = time.time()

        if letter != last_letter and (current_time - last_time) > delay:
            letters.append(letter)
            last_letter = letter
            last_time = current_time
            print("Detected:", letter)

    text_display = "".join(letters)
    cv2.putText(frame, text_display, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    annotated = results[0].plot()
    cv2.imshow("ASL Translator", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        letters.append(" ")
        last_letter = None
        print("Space added")

    if key == ord('d'):
        if len(letters) > 0:
            removed = letters.pop()
            print("Deleted:", removed)
        last_letter = None


    if key == ord('z'):
        letters = []
        last_letter = None
        print("Word cleared")


    if key == 13:
        final_text = "".join(letters)
        print("Final Text:", final_text)

        engine.say(final_text)
        engine.runAndWait()

        letters = []
        last_letter = None

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
