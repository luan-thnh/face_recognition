from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.student_data import StudentData

router = APIRouter()


@router.get("/checkin_history/")
async def get_checkin_history():
    try:
        df = StudentData.get_students_data()
        df = df.fillna('')
        result = []

        checkin_columns = [col for col in df.columns if col not in [
            'STT', 'ID', 'Họ và tên']]
        for _, row in df.iterrows():
            checkins = {date: str(row[date]) for date in checkin_columns}
            student_dict = {"id": str(row['ID']), "name": str(
                row['Họ và tên']), "checkins": checkins}
            result.append(student_dict)

        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
