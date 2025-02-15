import os
import cv2
import dlib
import pandas as pd
import numpy as np

# Initialize face detector and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to get face encodings
def get_face_encoding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = sp(gray, face)
        encoding = facerec.compute_face_descriptor(img, shape)
        return np.array(encoding)
    return None

def update_csv_with_encodings(image_folder, csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        print(f"{csv_file} not found.")
        return

    # Ensure 'Encoding' column exists
    if 'Encoding' not in df.columns:
        df['Encoding'] = None

    # Iterate through students and generate encodings for the images
    for index, row in df.iterrows():
        if pd.isna(row['Encoding']) or row['Encoding'] == '[]':
            student_id = str(row['ID'])  # Use student ID
            image_filename = f"{student_id}.jpg"  # Assuming the filename is the student ID
            image_path = os.path.join(image_folder, image_filename)

            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                encoding = get_face_encoding(img)

                if encoding is not None:
                    # Convert the encoding numpy array to a comma-separated string and store it
                    encoding_str = ','.join(map(str, encoding))  # Ensure the encoding is a string
                    df.at[index, 'Encoding'] = encoding_str
                    print(f"Encoding added for ID {student_id}")
                else:
                    print(f"No face detected in {image_filename}. Encoding will be skipped.")
            else:
                print(f"Image {image_filename} not found. Skipping student ID {student_id}.")

    # Save the updated CSV
    df.to_csv(csv_file, index=False)
    print(f"{csv_file} updated with encodings.")

# Main function to execute encoding generation
def main():
    image_folder = "student_images"  # Folder containing student images
    csv_file = "students.csv"  # CSV file to save the encodings
    update_csv_with_encodings(image_folder, csv_file)

# Run the main function
if __name__ == "__main__":
    main()
