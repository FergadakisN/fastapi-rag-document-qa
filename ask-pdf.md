Ask PDF
=======

Purpose
-------
Build a question-answering system based on information in a PDF file.

- Handling of PDF
- Handling long context with an LLM system.
- Expose functionality via a FastAPI Web service.

Background
----------
You are given

- A PDF file `Nielsen2025Natural-2026-03-20.pdf`

Extra processed PDF files

- Nielsen2025Natural-2026-03-20.pdf.tei.xml
- Nielsen2025Natural-2026-03-20.md


Task
----
Implement a FastAPI web service:

- with endpoint `POST /api/v1/ask` and JSON field `question`.
- returns answer in JSON with field `answer`



FastAPI Skeleton
----------------
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/api/v1/ask")
def ask(req: AskRequest):
    ...
    return {
        "answer": answer,
	"followup_questions": followup_questions,
    }

@app.post("/api/v1/upload-pdf")
def upload_pdf(...):
    # Optional
``` 


Example questions
-----------------
- What is the title
- Who is the author
- What are the two first sentences in the prompt engineering chapter
- What is the full reference given for Chain of thought prompting


Discussion
----------
- Which method would be most suitable for this task?
- What is the best conversion of the PDF?
- How can the PDF be split/chunked?


Requirements & Resources
------------------------
- Python
- FastAPI
- CampusAI LLM API
- Prompt engineering
  - Natural language processing - Chapter "Prompt engineering".
- Information retrieval
  - Natural language processing - Chapter "Information retrieval".
- Retrieval-augmented generation
  - Natural language processing - Chapter "Retrieval-augmented generation".
- ReAct
  - Natural language processing - Section 8.5 ReAct prompting.
  - Scientific literature 
    - SPINACH: SPARQL-Based Information Navigation for Challenging Real-World Questions
    - GRASP: Generic Reasoning And SPARQL Generation Across Knowledge Graphs


Deliverables
------------
A zipped repository in root (git archive -o latest.zip HEAD)
