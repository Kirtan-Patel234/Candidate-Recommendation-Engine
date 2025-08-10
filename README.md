# Candidate Recommendation Engine

This is a web app that recommends the best candidates for a given job description based on semantic relevance between the job and candidates' resumes. The app uses embedding models to rank candidates and provides AI-generated summaries explaining why a candidate is a good fit.

---

## Features

- Accepts a **job description** via text input
- Accepts **multiple candidate resumes** via PDF upload (supports multi-page resumes)
- Uses a fine-tuned **sentence transformer model** to generate vector embeddings
- Computes **cosine similarity** between job description and each resume
- Displays the **top 5 most relevant candidates** with their similarity scores
- Generates **AI-driven summaries** for each candidate explaining their fit using Google Gemini API
- Splits multi-resume PDFs into **individual candidate resumes** for easy review and download

---

## How It Works

1. **Embedding Model**  
   We use the [all-MiniLM-L6-v2](https://www.sbert.net/docs/pretrained_models.html) sentence transformer model as the base, fine-tuned on two public datasets for job description and resume matching:

   - `shamimhasan8/resume-vs-job-description-matching-dataset`
   - `surendra365/recruitment-dataset`

2. **PDF Resume Extraction**  
   Resumes are extracted from PDFs using PyMuPDF (`fitz`). The extractor intelligently groups multi-page resumes by detecting candidate names on pages. This helps handle cases where a resume spans multiple pages without repeating the name.

3. **Similarity Computation**  
   The job description and each resume are encoded into embeddings. Cosine similarity is computed to rank candidates by relevance.

4. **AI Summary Generation**  
   Using Google's Gemini-2.5-flash model (accessed via `google.generativeai`), the app generates a summary explaining:

   - Key strengths of the candidate
   - Gaps or missing qualifications
   - Suggested resume improvements
   - Overall hiring recommendation

5. **User Interface**  
   Built with Streamlit for an interactive, easy-to-use web experience.

---

## Setup & Installation

### Requirements

- Python 3.8+
- Streamlit
- PyMuPDF (`fitz`)
- sentence-transformers
- torch
- nameparser
- google-generativeai (Google Gemini API client)
- python-dotenv (for environment variable management)

### Installation

```bash
pip install streamlit fitz sentence-transformers torch spacy nameparser google-generativeai python-dotenv
```

or

```bash
pip install -r requirements.txt
```

### Usage

```bash
streamlit run app.py
```

## Model & Dataset Setup

This repository does **not** include the datasets and pretrained model files due to their large size. To run the app locally, please follow these steps:

### 1. Download the Datasets

You will need to download the datasets used for training the semantic similarity model from Kaggle:

- [Resume vs Job Description Matching Dataset](https://www.kaggle.com/shamimhasan8/resume-vs-job-description-matching-dataset)
- [Recruitment Dataset](https://www.kaggle.com/surendra365/recruitement-dataset)

Place the downloaded datasets in the `dataset/` directory of the project.

### 2. Train the Model

The app uses a fine-tuned SentenceTransformer model. To train or fine-tune the model:

- Use the provided training scripts in the `embedding/` folder.
- The training uses the datasets mentioned above.
- After training, save the fine-tuned model to the `model/all_minilm_finetuned/` directory.
- All the code is ready. You just have to run the embeddings.ipynb file.

### Further Improvements

1. Model performance and hardware constraints:
   Due to GPU memory limitations, I opted to use a lightweight SentenceTransformer model instead of a larger, more accurate one. With access to a more powerful system, the semantic similarity accuracy could be improved by leveraging a bigger or fine-tuned model.

2. Resume parsing limitations:
   The current resume extraction relies on simple heuristics and basic text parsing techniques, which may not perfectly handle complex or inconsistently formatted resumes. More advanced parsing methods, such as dedicated resume parsing APIs, or layout-aware processing, could significantly improve extraction accuracy and overall system reliability.

```

```
