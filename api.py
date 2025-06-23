from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import unicodedata
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fpdf import FPDF

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_path = "temp_uploaded.pdf"
storage = {}  # Used to persist vectors in-memory

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(pdf_path)
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_doc = text_splitter.split_documents(doc)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectors = chroma.Chroma.from_documents(final_doc, embeddings)

    storage['vectors'] = vectors
    return {"message": "File processed and vectors stored."}

@app.post("/generate")
async def generate_questions(
    topic: str = Form(...),
    difficulty_level: Literal["Easy", "Medium", "High"] = Form(...),
    question_type: Literal["MCQs", "Short Answer Questions", "Long Answer Questions"] = Form(...)
):
    if 'vectors' not in storage:
        return JSONResponse(status_code=400, content={"error": "No uploaded document found."})

    prompt = ChatPromptTemplate.from_template("""
You are a professional question paper setter.

**Input Document Content:**  
{context}

**User Inputs:**  
- Topic: {topic}  
- Difficulty Level: {difficulty_level}  
- Question Type: {question_type} (Choose from: MCQs, Short Answer Questions, Long Answer Questions)

### Your task:

1. Extract information related to the topic.
2. Generate **10 questions** relevant to the topic and matching the difficulty and question type.
3. If the question type is MCQs, include exactly 4 options (a–d) under each question.
4. Format your output so that questions and answers are clearly separated.

---

### Output Structure (Strictly Follow This):

**PDF Title: Generated {question_type} – Topic: {topic} – Level: {difficulty_level}**

#### Questions Section:
Q1. [Your question here]  
   a) Option 1  
   b) Option 2  
   c) Option 3  
   d) Option 4  
Q2. ...

---

#### Answers Section:
A1. [Correct option with full text, e.g., b) Neural Networks are inspired by the human brain.]
Do not include explanations or commentary.
""")

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="gemma2-9b-it"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = storage['vectors'].as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({
        "input": topic,
        "topic": topic,
        "difficulty_level": difficulty_level,
        "question_type": question_type
    })
    output_text = response['answer']

    questions, answers = [], []
    lines = output_text.splitlines()
    parsing_questions = True

    for line in lines:
        line = line.strip()
        if line == "---":
            continue
        if line.startswith("#### Answers Section"):
            parsing_questions = False
            continue
        if parsing_questions:
            if line.startswith("Q") or line.startswith(("a)", "b)", "c)", "d)")):
                questions.append(line)
        elif line.startswith("A"):
            answers.append(line)

    def clean_text(text):
        return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')

    def generate_pdf(content_list, title, is_question_pdf=True):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, clean_text(title), ln=True, align="C")
        pdf.ln(10)

        for line in content_list:
            line = line.strip()
            if is_question_pdf:
                if line.startswith("Q"):
                    pdf.set_font("Arial", 'B', 12)
                    pdf.multi_cell(0, 10, clean_text(line))
                elif line.startswith(("a)", "b)", "c)", "d)")):
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, "    " + clean_text(line))
            else:
                if line.startswith("A"):
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, clean_text(line))
            pdf.ln(2)

        return pdf.output(dest="S").encode("latin-1")

    q_title = f"Generated {question_type} – Topic: {topic} – Level: {difficulty_level}"
    a_title = f"Answers for {question_type} – Topic: {topic} – Level: {difficulty_level}"

    q_pdf = generate_pdf(questions, q_title, is_question_pdf=True)
    a_pdf = generate_pdf(answers, a_title, is_question_pdf=False)

    return {
        "questions_pdf": q_pdf.hex(),
        "answers_pdf": a_pdf.hex(),
        "time": round(time.process_time() - start, 2)
    }

@app.get("/ping")
def health():
    return {"status": "OK"}
