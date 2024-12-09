import pandas as pd


class StudentData:
    @staticmethod
    def get_students_data(file_path="excel_files/ST21A2A.xlsx"):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            raise Exception(f"Error reading student data: {e}")

    @staticmethod
    def update_checkin(df, student_id, checkin_column):
        try:
            student_row = df[df['ID'] == student_id]
            if not student_row.empty:
                if checkin_column not in df.columns:
                    df[checkin_column] = ''
                df.loc[df['ID'] == student_id, checkin_column] = 'X'
                return df
            else:
                raise ValueError("Student not found")
        except Exception as e:
            raise Exception(f"Error updating check-in: {e}")
