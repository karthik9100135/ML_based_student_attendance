import os
import cv2
import dlib
import pandas as pd
import numpy as np
from datetime import datetime
from db_helper import mark_student_attendance


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def mark_attendance(student_name, student_id, df):
    attendance_file = "attendance.xlsx"
    
    if not os.path.exists(attendance_file):
        print(f"Error: Attendance file {attendance_file} not found!")
        return

    df_attendance = pd.read_excel(attendance_file)
    current_time = datetime.now()

    if student_name not in df_attendance['Name'].values:
        new_entry = pd.DataFrame([[student_name, str(current_time)]], columns=['Name', 'Time'])
        df_attendance = pd.concat([df_attendance, new_entry], ignore_index=True)
        print(f"Attendance marked for {student_name}.")
        mark_student_attendance(student_name, student_id)  
    else:
        last_time = pd.to_datetime(df_attendance[df_attendance['Name'] == student_name]['Time'].iloc[0])

        if (current_time - last_time).seconds > 300:
            df_attendance.loc[df_attendance['Name'] == student_name, 'Time'] = str(current_time)
            print(f"Attendance time updated for {student_name}.")
            mark_student_attendance(student_name, student_id)  

    df_attendance.to_excel(attendance_file, index=False)

def detect_and_recognize_faces(student_dict):
    cap = cv2.VideoCapture(0)
    last_recognition_time = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            shape = sp(gray, face)
            encoding = facerec.compute_face_descriptor(frame, shape)
            encoding = np.array(encoding)

            matches = []
            for student_id,student_encoding in student_dict.items():
                distance = np.linalg.norm(student_encoding - encoding)
                if distance < 0.6:  
                    matches.append(student_id)

            if matches:
                student_id = matches[0]
                student_name = student_id 
                last_3_digits = student_id[-3:]
                current_time = datetime.now()

                if student_name not in last_recognition_time or (current_time - last_recognition_time[student_name]).seconds > 300:
                    mark_attendance(student_name, student_id, pd.read_excel("attendance.xlsx"))
                    last_recognition_time[student_name] = current_time
                    print(f"Face recognized as {student_name}")

                cv2.putText(frame, f"ID: {last_3_digits}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("Unknown face detected")

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    student_dict = {}
    df = pd.read_csv("studentsx.csv")

    for index, row in df.iterrows():
        encoding_str = row['Encoding']
        if isinstance(encoding_str, str) and encoding_str:
            try:
                student_dict[row['ID']] = np.array(list(map(float, encoding_str.strip('[]').split(','))))
            except ValueError:
                print(f"Invalid encoding for student ID {row['ID']}")
                continue
        else:
            print(f"Skipping student ID {row['ID']} due to missing or invalid encoding")

    detect_and_recognize_faces(student_dict)

if __name__ == "__main__":
    main()
