import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)
    try:
        for facial_landmark in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmark.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(frame, (x, y), 2, (0,0,255), -1)
    except Exception:
        pass

    cv2.imshow("Window", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
