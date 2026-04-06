import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
 
def extract_all_resumes(resume_text_list):
    """
    Takes a list of resumes (with 'file' and 'text') and returns structured JSON array
    using Gemini via LangChain.
    """
    print("Gemini_LLM is running")
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0
    )
 
    def extract_resume_data(resume_texts: str):
 
        prompt = f"""
You are an AI system that extracts structured information from resumes.
 
You may receive one or multiple resumes at a time.
 
Extract these fields from each resume:
 
- name
- skills:
    - technical
    - soft skills
- companies
- roles
- education (degree with - college)
 
Rules:
 
1. Return ONLY valid JSON.
2. Always return a JSON ARRAY, one object per resume.
3. Skills must be split into "technical" and "soft skills".
4. If a field is missing, use empty array [] or empty string "".
5. Do NOT include markdown, comments, or extra text.
 
JSON format example:
 
[
 {{
  "name": "",
  "skills": {{"technical": [], "soft": []}},
  "companies": [],
  "roles": [],
  "education": []
 }}
]
 
Resume Texts:
{resume_texts}
"""
 
        # Send prompt to Gemini using LangChain
        response = llm.invoke([HumanMessage(content=prompt)])
 
        text_output = response.content.strip()
 
        # Remove markdown if Gemini returns ```json
        if text_output.startswith("```"):
            lines = text_output.split("\n")
            text_output = "\n".join(lines[1:-1]).strip()
 
        # Parse JSON safely
        try:
            return json.loads(text_output)
        except json.JSONDecodeError:
            print("Gemini returned invalid JSON:")
            print(text_output)
            return []
 
    # ---------------- Process All Resumes ----------------
    all_resumes = []
 
    for resume in resume_text_list:
        extracted = extract_resume_data(resume["text"])
        all_resumes.extend(extracted)
 
    # Pretty print final JSON array
    print(json.dumps(all_resumes, indent=2))
    return all_resumes
 
 
 

 
 