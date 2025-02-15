import mysql.connector
from datetime import datetime

def mark_student_attendance(student_name, student_id):
    # Establish connection to MySQL database
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Shankar@0115',
            database='attendance_db'
        )
        cursor = conn.cursor()

        # Debug: Print the received values
        print(f"Marking attendance for Name: {student_name}, ID: {student_id}")

        # Get current time to determine the hour column
        current_time = datetime.now().time()
        if current_time >= datetime.strptime("09:40", "%H:%M").time() and current_time < datetime.strptime("10:30", "%H:%M").time():
            hour_column = 'Hour1'
        elif current_time >= datetime.strptime("10:30", "%H:%M").time() and current_time < datetime.strptime("11:20", "%H:%M").time():
            hour_column = 'Hour2'
        elif current_time >= datetime.strptime("11:20", "%H:%M").time() and current_time < datetime.strptime("12:10", "%H:%M").time():
            hour_column = 'Hour3'
        elif current_time >= datetime.strptime("12:10", "%H:%M").time() and current_time < datetime.strptime("13:00", "%H:%M").time():
            hour_column = 'Hour4'
        elif current_time >= datetime.strptime("13:50", "%H:%M").time() and current_time < datetime.strptime("14:40", "%H:%M").time():
            hour_column = 'Hour5'
        elif current_time >= datetime.strptime("19:40", "%H:%M").time() and current_time < datetime.strptime("20:50", "%H:%M").time():
            hour_column = 'Hour6'
        else:
            print("Outside attendance marking hours.")
            return

        # Check if the student already exists in the table
        check_query = "SELECT student_id FROM attendance WHERE student_id = %s"
        cursor.execute(check_query, (student_id,))
        result = cursor.fetchone()

        if result:
            print(f"Student {student_id} already exists in the database.")
            # Check if attendance is already marked for the current hour
            check_hour_query = f"SELECT {hour_column} FROM attendance WHERE student_id = %s"
            cursor.execute(check_hour_query, (student_id,))
            hour_result = cursor.fetchone()

            if hour_result and hour_result[0] == 1:
                print(f"Attendance for {student_name} (ID: {student_id}) already marked for {hour_column}.")
            else:
                # Update attendance for the current hour
                update_query = f"UPDATE attendance SET {hour_column} = %s WHERE student_id = %s"
                cursor.execute(update_query, (1, student_id))
                conn.commit()
                print(f"Attendance marked for {student_name} (ID: {student_id}) in {hour_column}.")
        else:
            # Insert new student record and mark attendance for the current hour
            insert_query = f"INSERT INTO attendance (student_id, student_name, {hour_column}) VALUES (%s, %s, %s)"
            print(f"Executing insert query: {insert_query} with values ({student_id}, {student_name}, 1)")
            cursor.execute(insert_query, (student_id, student_name, 1))
            conn.commit()
            print(f"New entry created and attendance marked for {student_name} (ID: {student_id}) in {hour_column}.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()

