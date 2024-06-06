from fastapi import FastAPI, File, UploadFile, HTTPException 
from pydantic import BaseModel
from process import process_pdf, answer_question 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# for COROS
origins = [
    "http://ai-planet:65519",  
    "http://ai-planet:65520",  
]
# Allow access
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
extracted_text = ""

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global extracted_text
    result = await process_pdf(file)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    extracted_text = result["text"]
    
    return {"message": "PDF processed successfully"}

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No PDF content available. Please upload a PDF first.")
    answer = answer_question(extracted_text, request.question)
    return {"answer": answer}
