import re
import string
import streamlit as st
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text, model):
    return model.encode(text)

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)

def score_resumes(resume_texts, job_description):
    if not resume_texts:
        return []
    
    model = load_embedding_model()
    job_desc_embedding = get_embedding(preprocess_text(job_description), model)
    resume_embeddings = [get_embedding(preprocess_text(resume), model) for resume in resume_texts]
    
    similarity_scores = [cosine_similarity([job_desc_embedding], [resume_emb])[0][0] for resume_emb in resume_embeddings]
    similarity_scores = [round(score * 100, 2) for score in similarity_scores]
    
    job_keywords = extract_keywords(job_description)
    resume_keyword_counts = [len(extract_keywords(resume) & job_keywords) for resume in resume_texts]
    
    ranked_resumes = sorted(zip(similarity_scores, resume_keyword_counts, resume_texts), key=lambda x: (-x[0], -x[1]))
    return ranked_resumes

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text else "No text extracted from PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def main():
    st.title("AI-Powered Resume Screening and Ranking System")
    st.write("Upload one or more resumes (TXT or PDF) to check their match with the job description.")
    
    job_description = st.text_area("Enter the job description:", "")
    
    uploaded_files = st.file_uploader("Upload resumes", type=["txt", "pdf"], accept_multiple_files=True)
    
    if uploaded_files and job_description.strip():
        resume_texts = []
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    if extracted_text and extracted_text != "No text extracted from PDF.":
                        resume_texts.append(extracted_text)
                else:
                    resume_texts.append(uploaded_file.read().decode("utf-8"))
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        if resume_texts:
            ranked_resumes = score_resumes(resume_texts, job_description)
            for i, (score, keyword_count, resume) in enumerate(ranked_resumes):
                st.write(f"Resume {i+1} matches the job description by {score}%. Keyword Matches: {keyword_count}")
        else:
            st.warning("No valid resumes were processed.")
    
if __name__ == "__main__":
    main()


