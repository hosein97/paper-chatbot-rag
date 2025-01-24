from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.services import process_and_store_file, process_question

router = APIRouter()

@router.post("/upload", summary="Upload a PDF file")
async def upload_file(file: UploadFile = File(...)):
    print(file)
    # Validate file extension based on filename
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")    
    try:
        filename = await process_and_store_file(file)
        return {"filename": filename, "message": "File uploaded and stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/chat", response_model=ChatResponse, summary="Chat with uploaded PDF")
async def chat_with_paper(request: ChatRequest):
    answer = process_question(request.filename, request.question)
    if not answer:
        raise HTTPException(status_code=404, detail="Unable to generate an answer")
    return {"answer": answer}

