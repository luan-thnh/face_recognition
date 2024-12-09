from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.frame_processing import process_video

router = APIRouter()


@router.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    try:
        return await process_video(file)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
