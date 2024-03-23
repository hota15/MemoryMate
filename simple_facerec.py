import cv2
import face_recognition
import os
import glob
import numpy as np
from datetime import datetime
import uuid

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing
        self.frame_resizing = 0.25
        self.unknown_count = 0
        self.capture_image = False

    def load_encoding_images(self, images_path):
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            # Get encoding if faces are found in the image
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                img_encoding = face_encodings[0]

                # Store file name and file encoding
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            else:
                print(f"No face found in {filename}, skipping.")

        print("Encoding images loaded")


    def detect_known_faces(self, frame, consecutive_frames=10):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        if all(name == "Unknown" for name in face_names):
            # If all detected faces are classified as "Unknown" for consecutive frames, capture the image of the person
            self.unknown_count += 1

            if self.unknown_count >= consecutive_frames and not self.capture_image:
                # Capture only the area within the first detected face rectangle
                y1, x2, y2, x1 = face_locations[0].astype(int)
                face_region = frame[y1:y2, x1:x2]

                # Save the captured face region as an image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unknown_person_path = os.path.join("C:\\Users\\tejas\\Desktop\\Hashcode\\images", f"unknown_person_{timestamp}.jpg")
                cv2.imwrite(unknown_person_path, face_region)
                print(f"Unknown person face region captured: {unknown_person_path}")
                self.capture_image = True
        else:
            # Reset the counters if a known face is detected
            self.unknown_count = 0
            self.capture_image = False

        return face_locations.astype(int), face_names
    
# Example usage:
if __name__ == "__main__":
    
    
    sfr = SimpleFacerec()
    sfr.load_encoding_images("C:\\Users\\tejas\\Desktop\\Hashcode\\images")
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        face_locations, face_names = sfr.detect_known_faces(frame, consecutive_frames=10)

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
