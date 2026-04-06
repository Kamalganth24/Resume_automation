# main.py
from resume_processing import main as process_resumes
from embed import build_index

def control_flow():
    print("Control got from monitor & sent to resume_processing")
    resume_text_list = process_resumes()
    print("Resumes extracted:", resume_text_list)
    index, entries = build_index(resume_text_list)
    print("FAISS index built with", index.ntotal, "vectors")
    return index, entries
