from io import BytesIO
import streamlit as st
import os
import fitz
from nameparser import HumanName
from sentence_transformers import SentenceTransformer, util
import torch
import re
import google.generativeai as genai
from dotenv import load_dotenv
from nameparser import HumanName


# load in google api key to generate resume summaries
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# load finetuned model on CPU first (to avoid meta tensor error)
model = SentenceTransformer(
    "model/all_minilm_finetuned", device="cpu")
model = model.to(device)

def normalize_name(name):
    return " ".join(name.split()).lower()


# extract text from PDF (multiple if needed)
# this function splits lists of resumes by name

def extract_resumes_with_pages(pdf_bytes):
    resumes = []
    current_resume = {"name": None, "text": "", "pages": []}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=0):
            text = page.get_text()
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            name = None
            if lines:
                name_obj = HumanName(lines[0])
                possible_name = f"{name_obj.first} {name_obj.middle} {name_obj.last}".strip()
                if name_obj.first and name_obj.last:
                    name = possible_name

            if name:
                if name != current_resume["name"]:
                    # Save previous resume
                    if current_resume["name"] is not None:
                        # Extract PDF bytes for previous resume pages
                        pdf_bytes_io = BytesIO()
                        new_doc = fitz.open()
                        for p in current_resume["pages"]:
                            new_doc.insert_pdf(doc, from_page=p, to_page=p)
                        new_doc.save(pdf_bytes_io)
                        new_doc.close()

                        current_resume["pdf_bytes"] = pdf_bytes_io.getvalue()
                        resumes.append(current_resume)

                    # Start new resume
                    current_resume = {
                        "name": name,
                        "text": text,
                        "pages": [page_num]
                    }
                else:
                    # Same resume continues
                    current_resume["text"] += "\n" + text
                    current_resume["pages"].append(page_num)
            else:
                # no name found, treat as continuation if possible
                if current_resume["name"] is not None:
                    current_resume["text"] += "\n" + text
                    current_resume["pages"].append(page_num)
                else:
                    # No current resume? start unknown
                    current_resume = {
                        "name": "Unknown",
                        "text": text,
                        "pages": [page_num]
                    }

        # Add last resume
        if current_resume["name"] is not None and current_resume["text"]:
            pdf_bytes_io = BytesIO()
            new_doc = fitz.open()
            for p in current_resume["pages"]:
                new_doc.insert_pdf(doc, from_page=p, to_page=p)
            new_doc.save(pdf_bytes_io)
            new_doc.close()

            current_resume["pdf_bytes"] = pdf_bytes_io.getvalue()
            resumes.append(current_resume)

    return resumes



# used to get name from resume (using nameparser)
def guess_name_from_text(text):
    # take only first 10 lines (or less) to avoid catching unrelated headers
    lines = text.strip().split('\n')[:10]
    
    # filter lines that are too short or likely not names
    candidate_lines = [line.strip() for line in lines if 2 <= len(line.split()) <= 4]

    for line in candidate_lines:
        # check if line looks like a name (e.g. contains letters, no digits)
        if re.match(r'^[A-Za-z .\-]+$', line):
            name = HumanName(line)
            # check if it parsed at least first and last name
            if name.first and name.last:
                return name.full_name
    return "Unknown"
    
# preprocess text for better accuracy

def clean_resume_text(text):
    text = text.lower()
    # collapse multiple spaces/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# compute cosine similarity using embeddings


def compute_similarity(job_text, resume_text):
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity_score = util.cos_sim(job_embedding, resume_embedding)
    return similarity_score.item()

# generate resume summary using google gemini model


def get_gemini_response(prompt):
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text


# STREAMLIT UI

# set page config
st.set_page_config(page_title="Candidate Recommendation Engine")

st.set_page_config(page_title="Candidate Recommendation Engine",
                   page_icon="🔍", layout="wide")

# app title
st.markdown("<h1 style='text-align: center;'>🔍 Candidate Recommendation Engine</h1>",
            unsafe_allow_html=True)

# job description input
st.subheader("Job Description")
job_description = st.text_area("Paste the job description below:",
                               height=180, placeholder="Enter job description here...")

# file uploader
st.subheader("Upload Candidate Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF resumes (single or multiple)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and job_description.strip():
    candidates = []

    for file in uploaded_files:
        pdf_bytes = file.getvalue()
        resumes = extract_resumes_with_pages(pdf_bytes)
        for idx, resume in enumerate(resumes, start=1):
            cleaned_text = clean_resume_text(resume["text"])
            candidate_name = resume["name"] or guess_name_from_text(resume["text"])
            sim_score = compute_similarity(job_description, cleaned_text)
            candidates.append({
                "name": candidate_name,
                "score": sim_score,
                "text": resume["text"],
                "file_bytes": resume["pdf_bytes"],  # individual PDF bytes
                "file_name": f"{candidate_name.replace(' ', '_')}_resume.pdf"
            })

    # sort by similarity
    top_candidates = sorted(
        candidates, key=lambda x: x["score"], reverse=True)[:5]

    st.markdown("## 🏆 Top 5 Candidates")

    for idx, candidate in enumerate(top_candidates, start=1):
        with st.container():
            st.markdown(f"### {idx}. {candidate['name']}")

            # score progress bar
            progress_value = max(0.0, min(candidate['score'], 1.0))
            st.progress(progress_value)
            st.write(f"**Similarity Score:** {candidate['score']*100:.2f}%")

            st.download_button(
                label=f"📄 View {candidate['name']}'s Resume",
                data=candidate['file_bytes'],
                file_name=candidate['file_name'],
                mime="application/pdf",
                key=f"download_{idx}_{candidate['name'].replace(' ', '_')}"
            )

            # add unique key so each button is unique
            if st.button(f"Why is this candidate a good fit?", key=f"fit_button_{idx}"):
                with st.spinner("Generating resume summary..."):
                    # create prompt to give to google gemini model
                    resume_summary_prompt = f"You are an experienced recruiter and hiring manager. I will provide you with a job description and a candidate's resume.Your task is to compare the resume to the job description and give me the following: Key Strengths – list the most relevant skills, experiences, and achievements the candidate has that align with the role. Gaps or Missing Qualifications – identify skills, certifications, or experiences from the job description that are not evident in the resume. Suggested Improvements – recommend specific changes or additions to the resume to make it a stronger match for the job. Don't include resume format tips in here, but just focus on skills the candidate could improve. Overall Recommendation – brief hiring recommendation (e.g., strong fit, moderate fit, weak fit). Please do NOT include introductory phrases such as Of course or references to yourself. Start directly with the analysis. Here is the job description: {
                        job_description}. Here is the resume: {candidate['text']}. You can use the candidate's name find in the resume if you want. Make sure to be critical of the resumes."

                    # generate resume summary and print when button clicked
                    response = get_gemini_response(resume_summary_prompt)
                    st.write(response)

            st.markdown("---")

else:
    st.info("Please enter a job description and upload at least one resume.")
