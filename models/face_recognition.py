from deepface import DeepFace
import cv2

IMG_PATH = "img3.jpg"

detected_faces = DeepFace.extract_faces(img_path=IMG_PATH, align=True, detector_backend='retinaface')
face_data = detected_faces[0]['facial_area']
print(face_data)

# Bounding Box Placement
cv_img = cv2.imread(IMG_PATH)

x1, y1, width, height, left_eye, right_eye = face_data.values()

START_POINT = (x1, y1)
END_POINT = (x1 + width, y1 + height)
GREEN = (0, 255, 0) # Color in BGR format
THICKNESS = 4 # in pixels

face_with_box = cv2.rectangle(cv_img, START_POINT, END_POINT, GREEN, THICKNESS)

# Save the image with the bounding box
cv2.imwrite("processed_" + IMG_PATH, face_with_box)