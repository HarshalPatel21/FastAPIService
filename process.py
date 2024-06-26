import fitz  # PyMuPDF
from fastapi import UploadFile 
from transformers import pipeline
from transformers import BartForConditionalGeneration ,BartTokenizer ,AutoTokenizer,AutoModelForQuestionAnswering
import torch
import os

# variables 

model_name = "deepset/roberta-base-squad2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pipelines
qa_pipeline = pipeline("question-answering",model=model_name,tokenizer=model_name)
text_gen_pipeline = pipeline("text-generation", model="gpt2",tokenizer="gpt2")

summarization_model_name = "facebook/bart-large-cnn"
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name).to(device)
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarizer = pipeline("summarization", model=summarization_model, tokenizer=summarization_tokenizer, device=0 if torch.cuda.is_available() else -1)

model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# add token padding
if tokenizer.pad_token is None :
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


async def process_pdf(file: UploadFile):

    # check for PDF
    if file.content_type != "application/pdf":
        return {"error": "File must be a PDF"}
    
    content = await file.read()
    pdf_document = fitz.open(stream=content, filetype="pdf")

    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()

    # if text becomes too big we need to summarize it
    if len(text) > 5000:  
        text = summarize_text(text)

    return {"text": text}



def answer_question(context: str, question: str) -> str:
    # get answer from qa pipeline
    answer = qa_pipeline(question=question, context=context)['answer']

    # expand the answer
    expanded_answer = text_gen_pipeline(f"{question} answer is : {answer}", max_length=100, num_return_sequences=1,do_sample=True)[0]['generated_text']

    return expanded_answer


def summarize_text(text, max_length=150, min_length=50):
    
    max_length = 512  
    min_length = 50  

    # Break the text into chunks
    chunk_size = 1024
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    summary = ""
    for chunk in text_chunks:
        # Summarize each chunk
        summarized_chunk = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summary += summarized_chunk + " "

    return summary.strip()