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
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = sp(gray, face)
            print(shape)
            encoding = facerec.compute_face_descriptor(img, shape)
            return np.array(encoding)
    except Exception as e:
        print(f"Error during face encoding: {e}")
    return None

# Function to update CSV with encodings
def update_csv_with_encodings(image_folder, csv_file):
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found.")
        return

    df = pd.read_csv(csv_file)

    # Ensure 'Encoding' column exists
    if 'Encoding' not in df.columns:
        df['Encoding'] = None

    # Logging for missing or failed images
    missing_images = []
    no_face_detected = []

    # Iterate through students and generate encodings for the images
    for index, row in df.iterrows():
        student_id = str(row['ID'])  # Use student ID
        image_filename = f"{student_id}.jpg"  # Assuming the filename is the student ID
        image_path = os.path.join(image_folder, image_filename)

        if pd.isna(row['Encoding']) or row['Encoding'] == '[]':
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                encoding = get_face_encoding(img)

                if encoding is not None:
                    # Convert the encoding numpy array to a comma-separated string and store it
                    encoding_str = ','.join(map(str, encoding))
                    df.at[index, 'Encoding'] = encoding_str
                    print(f"Encoding added for ID {student_id}")
                else:
                    print(f"No face detected in {image_filename}. Encoding will be skipped.")
                    no_face_detected.append(student_id)
            else:
                print(f"Image {image_filename} not found. Skipping student ID {student_id}.")
                missing_images.append(student_id)

    # Save the updated CSV
    df.to_csv(csv_file, index=False)
    print(f"CSV file '{csv_file}' updated with encodings.")

    # Log issues
    if missing_images:
        print(f"\nMissing images for student IDs: {missing_images}")
    if no_face_detected:
        print(f"\nNo face detected in images for student IDs: {no_face_detected}")

# Main function to execute encoding generation
def main():
    image_folder = "student_images"  # Folder containing student images
    csv_file = "studentsx.csv"  # CSV file to save the encodings
    update_csv_with_encodings(image_folder, csv_file)

# Run the main function
if __name__ == "__main__":
    main()
