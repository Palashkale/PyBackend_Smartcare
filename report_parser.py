import fitz 
import docx
import re


def load_symptoms_from_training_data(path="./Dataset/training.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    return df.columns.drop("prognosis").tolist()

SYMPTOM_LIST = load_symptoms_from_training_data()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Clean and match symptoms
def extract_symptoms_from_text(text):
    text = text.lower()
    found_symptoms = []
    for symptom in SYMPTOM_LIST:
        pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
        if re.search(pattern, text):
            found_symptoms.append(symptom)
    return found_symptoms
