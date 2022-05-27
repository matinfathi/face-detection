from tensorflow.keras.models import load_model
from utils import read_config, crop_images
import cv2

config = read_config()

# Read constant variables
SAVE_MODEL_PATH = config["save_model_path"]
TRUE_POSITIVE = (0, 255, 0)
TRUE_NEGATIVE = (0, 0, 255)
CLASS_NAMES = ['matin', 'not_matin']

# Define cv2 face localization
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load model and define classes
model = load_model(SAVE_MODEL_PATH)

# Turn on web camera
video_capture = cv2.VideoCapture(0)

print("Streaming started - to quit press ESC")
while True:

    # Capture each frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # for each face on the image detected by OpenCV
        # get extended image of this face
        face_image = crop_images(frame, x, y, w, h, 0.8)

        # classify face and draw a rectangle around the face
        # green for positive class and red for negative
        result = model.predict(face_image, verbose=0)
        prediction = CLASS_NAMES[round(result[0][0])]  # predicted class

        if prediction == 'matin':
            color = TRUE_POSITIVE
            confidence = 1 - result[0][0]
        else:
            color = TRUE_NEGATIVE
            confidence = result[0][0]
        # draw a rectangle around the face
        cv2.rectangle(frame,
                      (x, y),  # start_point
                      (x+w, y+h),  # end_point
                      color,
                      2)  # thickness in px
        cv2.putText(frame,
                    # text to put
                    "{:6} - {:.2f}%".format(prediction, confidence*100),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,  # font
                    2,  # fontScale
                    color,
                    2)  # thickness in px

    # display the resulting frame
    cv2.imshow("Face detector - to quit press ESC", frame)

    # Exit with ESC
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC code
        break


# when everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Streaming ended")
