# 🎓 University Admission Assistant AI

AI-powered assistant that helps students in Kazakhstan find suitable university programs based on their exam score, field of study, city, and profile subjects.

The system combines a PostgreSQL database with an AI chat interface to provide accurate admission suggestions.

---

## Features

- Natural language chat interface
- Automatic extraction of:
  - Exam score
  - Field of study
  - City preference
  - Profile subjects
- Database-driven recommendations (no fake data)
- Session-based conversation memory
- REST API for mobile/web apps
- CLI version for local testing

---

## Tech Stack

- Python
- FastAPI
- PostgreSQL
- OpenAI API
- Psycopg
- Pydantic
- Uvicorn

---

## Project Structure



AiChat/
│
├── server.py # FastAPI backend
├── cli_app.py # Console assistant
├── keys.env # Environment variables (not uploaded)
├── README.md
└── requirements.txt



---

## Installation

Clone repository:

```bash
git clone https://github.com/kuatsaparali/university-admission-assistant-AI-.git
cd university-admission-assistant-AI-



Run Server
Start API server:
uvicorn server:app --reload
Server runs at:
http://127.0.0.1:8000



API Example
Request
POST /chat
{
  "session_id": "user1",
  "text": "I scored 95 and want IT in Almaty"
}
Response
{
  "reply": "...assistant message...",
  "results_count": 5
}


🔄 Reset Session
DELETE /reset/{session_id}


📱 Integration
The API can be used by:
iOS apps (Swift)
Android apps
Web frontends
Chatbots


🔮 Future Improvements
Persistent conversation storage
Scholarship suggestions
Program ranking
Multi-language support
Admin dashboard


👨‍💻 Author
Kuat Saparaly
Astana IT University student project.
