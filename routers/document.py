from fastapi import APIRouter, UploadFile
from fastapi.params import File

router = APIRouter(prefix='/documents', tags=['Документы'])


@router.post("")
async def upload_document(file: UploadFile = File(...)):
    ...
@router.get("")
async def get_info_from_document():
    ...