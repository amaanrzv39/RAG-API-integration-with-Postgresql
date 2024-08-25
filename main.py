from fastapi import FastAPI, UploadFile, HTTPException, Depends, BackgroundTasks
import os
import shutil
import io
from db import get_db, File, FileChunk
from sqlalchemy.orm import Session
from file_parser import FileParser
from background_tasks import TextProcessor, client
from sqlalchemy import select
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


class QuestionModel(BaseModel):
    question: str

class AskModel(BaseModel):
    document_id: int
    question: str


@app.get("/")
async def root(db:Session = Depends(get_db)):
    files_query = select(File)
    files = db.scalars(files_query).all()
    files_list = [{"file_id":file.file_id, "file_name":file.file_name} for file in files]
    return files_list


@app.post("/uploadfile/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile, db: Session = Depends(get_db)):
    allowed_extensions = ["txt", "pdf"]
    file_extension = file.filename.split('.')[-1]
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not allowed")

    folder = "Files"
    try:
        os.makedirs(folder, exist_ok=True)
        file_location = os.path.join(folder, file.filename)
        file_content = await file.read()
        with open(file_location, "wb+") as fdest:
            fsrc = io.BytesIO(file_content)
            shutil.copyfileobj(fsrc, fdest)

        content_parser = FileParser(file_location)
        file_text_content = content_parser.parse()
        new_file = File(file_name=file.filename, file_content=file_text_content)
        
        db.add(new_file)
        db.commit()
        db.refresh(new_file)

        # Add background job for processing file content
        background_tasks.add_task(TextProcessor(db, new_file.file_id).chunk_and_embed, file_text_content)
        
        return {"info": "File saved", "filename":file.filename}
    except Exception as e:
        print(f"Error saving the file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")

async def get_similar_chunks(file_id: int, question: str, db: Session):
    try:
        response = client.embeddings.create(input=question, model="text-embedding-ada-002")
        question_embedding = response.data[0].embedding

        similar_chunks_query = select(FileChunk).where(FileChunk.file_id==file_id).order_by(FileChunk.embedding_vector.l2_distance(question_embedding)).limit(10)
        similar_chunks = db.scalars(similar_chunks_query).all()
        return similar_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/")
async def ask_question(request: AskModel, db:Session = Depends(get_db)):
    if OPENAI_API_KEY is None:
        raise HTTPException(status_code=500, detail="API key missing")
    try:
        similar_chunks = await get_similar_chunks(request.document_id, request.question, db)
        context_texts = [chunk.chunk_text for chunk in similar_chunks]
        context = " ".join(context_texts)

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content":f"You are a helpful assistant. Here is the context to use to reply to questions: {context}"},
                {"role":"system", "content":request.question}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/find-similar-chunks/{file_id}")
async def find_similar_chunks_endpoint(file_id: int, question_data: QuestionModel, db: Session = Depends(get_db)):
    try:
        similar_chunks = await get_similar_chunks(file_id, question_data.question, db)

        list_of_chunks = [
            {"chunk_id":chunk.chunk_id, "chunk_text":chunk.chunk_text}
            for chunk in similar_chunks
        ]
        
        return list_of_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

    