from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.langchain_service import process_question, clear_memory
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_manager import process_and_store_file

router = APIRouter()

@router.post("/upload", summary="Upload a PDF file")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        file_id = await process_and_store_file(file)
        clear_memory()
        return {"file_id": file_id, "message": "File uploaded and stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/chat", response_model=ChatResponse, summary="Chat with uploaded PDF")
async def chat_with_paper(request: ChatRequest):
    answer = process_question(request.file_id, request.question)
    if not answer:
        raise HTTPException(status_code=404, detail="Unable to generate an answer")
    return {"answer": answer}

@router.post("/new_chat", summary="Start a new chat")
async def start_new_chat():
    """Clear the chat history for a new chat session."""
    clear_memory()  # Reset memory for a new chat
    return {"message": "New chat started, chat history cleared."}
