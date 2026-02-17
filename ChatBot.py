import os
import re
import json
from typing import Optional, List, Dict, Any
from collections import defaultdict

from dotenv import load_dotenv
from psycopg import connect
from openai import OpenAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================
# Config
# =========================
load_dotenv("keys.env")

DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

if not DB_URL:
    raise RuntimeError("DATABASE_URL not found in keys.env")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in keys.env")

client = OpenAI(api_key=API_KEY)

MAX_CONTEXT_MESSAGES = 20  # memory per session_id (only in RAM)
SYSTEM_PROMPT = """
You are a friendly university admission assistant in Kazakhstan.
Speak like a helpful friend: warm, simple, not formal.

You receive:
- user's message
- extracted filters (score/field/city/profile subjects) if we could detect them
- database results (programs)

Rules:
- Use ONLY the database results. Do not invent universities or min scores.
- If score is missing, ask ONE short question to get it.
- If field is missing, ask what specialty they want (e.g., IT, Law, Medical, Engineering).
- If city is missing, you can show options from all cities and ask if they prefer a city.
- If profile subjects are missing, ask which profile subjects (e.g., Math+Physics, Math+Informatics).
- If there are matches, show 3–6 best options as bullet points.
- Always mention min_score and whether the user passes.
- Give 1–3 practical next steps (documents, applying, etc.)
Keep it concise.
""".strip()


# =========================
# Helpers / Extractors
# =========================
def clean_user_input(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*you:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_score(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{2,3})\b", text)
    if not m:
        return None
    score = int(m.group(1))
    if score < 0 or score > 200:
        return None
    return score


def extract_city(text: str) -> Optional[str]:
    t = text.lower()
    city_map = [
        (["astana", "астана"], "Astana"),
        (["almaty", "алматы"], "Almaty"),
        (["kaskelen", "каскелен"], "Kaskelen"),
        (["karaganda", "караганда"], "Karaganda"),
        (["shymkent", "шымкент"], "Shymkent"),
        (["turkistan", "туркестан"], "Turkistan"),
    ]
    for variants, city in city_map:
        for v in variants:
            if v in t:
                return city
    return None


def extract_field(text: str) -> Optional[str]:
    t = text.lower()
    field_map = [
        (["it", "айти", "информатика", "computer science", "cs"], "IT"),
        (["cyber", "кибер", "кибербезопас"], "Cyber Security"),
        (["law", "юрист", "право"], "Law"),
        (["medicine", "medical", "медицина", "врач"], "Medical"),
        (["engineering", "инженер", "техн"], "Engineering"),
        (["tourism", "туризм"], "Tourism"),
        (["telecom", "телеком", "связь"], "Telecommunication"),
    ]
    for keywords, field in field_map:
        for k in keywords:
            if k in t:
                return field
    return None


def extract_profile_subjects(text: str) -> Optional[str]:
    t = text.lower()

    if (("физ" in t) and ("мат" in t)) or (("physics" in t) and ("math" in t)):
        return "Mathematics + Physics"
    if (("информ" in t) and ("мат" in t)) or (("informatics" in t) and ("math" in t)):
        return "Mathematics + Informatics"
    if (("биол" in t) and ("хим" in t)) or (("biology" in t) and ("chem" in t)):
        return "Biology + Chemistry"

    return None


def recover_from_history(history: List[Dict[str, str]]) -> Dict[str, Any]:
    last_score = None
    last_city = None
    last_field = None
    last_subjects = None

    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")

        if last_score is None:
            s = extract_score(content)
            if s is not None:
                last_score = s
        if last_city is None:
            c = extract_city(content)
            if c is not None:
                last_city = c
        if last_field is None:
            f = extract_field(content)
            if f is not None:
                last_field = f
        if last_subjects is None:
            ps = extract_profile_subjects(content)
            if ps is not None:
                last_subjects = ps

        if last_score and last_city and last_field and last_subjects:
            break

    return {
        "score": last_score,
        "city": last_city,
        "field": last_field,
        "profile_subjects": last_subjects
    }


# =========================
# DB
# =========================
def get_view_columns(conn) -> List[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'programs_with_university'
    ORDER BY ordinal_position;
    """
    with conn.cursor() as cur:
        cur.execute(q)
        return [r[0] for r in cur.fetchall()]


def fetch_programs(score: int, field: Optional[str], city: Optional[str],
                   profile_subjects: Optional[str], limit: int = 8) -> List[Dict[str, Any]]:
    with connect(DB_URL) as conn:
        cols = set(get_view_columns(conn))

        program_expr = "program_name" if "program_name" in cols else "program_code AS program_name"
        has_subjects = "profile_subjects" in cols
        subjects_expr = "profile_subjects" if has_subjects else "NULL::text AS profile_subjects"

        subjects_filter = ""
        if has_subjects:
            subjects_filter = "AND (%(profile_subjects)s::text IS NULL OR lower(profile_subjects) = lower(%(profile_subjects)s::text))"

        query = f"""
        SELECT
            university_name,
            city,
            {program_expr},
            field,
            language,
            duration_years,
            min_score,
            degree_level,
            {subjects_expr}
        FROM programs_with_university
        WHERE min_score <= %(score)s
          AND (%(field)s::text IS NULL OR lower(field) = lower(%(field)s::text))
          AND (%(city)s::text  IS NULL OR lower(city)  = lower(%(city)s::text))
          {subjects_filter}
        ORDER BY min_score DESC
        LIMIT %(limit)s;
        """

        params = {
            "score": score,
            "field": field,
            "city": city,
            "profile_subjects": profile_subjects,
            "limit": limit,
        }

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [
        {
            "university": r[0],
            "city": r[1],
            "program": r[2],
            "field": r[3],
            "language": r[4],
            "duration_years": r[5],
            "min_score": r[6],
            "degree_level": r[7],
            "profile_subjects": r[8],
        }
        for r in rows
    ]


# =========================
# OpenAI
# =========================
def ask_ai(history: List[Dict[str, str]], extracted: Dict[str, Any], db_results: List[Dict[str, Any]]) -> str:
    payload = {
        "extracted": extracted,
        "db_results": db_results,
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_CONTEXT_MESSAGES:])
    messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.5,
        messages=messages,
    )
    return resp.choices[0].message.content


# =========================
# FastAPI server
# =========================
app = FastAPI()

# Чтобы не было проблем, когда Swift обращается к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Память только в RAM, пока сервер включен
session_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)


class ChatIn(BaseModel):
    session_id: str
    text: str


class ChatOut(BaseModel):
    reply: str
    extracted: Dict[str, Any]
    results_count: int


@app.get("/")
def home():
    return {"message": "Server works"}


@app.post("/chat", response_model=ChatOut)
def chat_endpoint(data: ChatIn):
    session_id = data.session_id.strip()
    text = clean_user_input(data.text)

    if not session_id:
        return {"reply": "❌ session_id is required", "extracted": {}, "results_count": 0}
    if not text:
        return {"reply": "Напиши сообщение 🙂", "extracted": {}, "results_count": 0}

    history = session_histories[session_id]

    # add user msg to history
    history.append({"role": "user", "content": text})

    # extract from current message
    score = extract_score(text)
    field = extract_field(text)
    city = extract_city(text)
    profile_subjects = extract_profile_subjects(text)

    # recover from history (this session only)
    recovered = recover_from_history(history)
    if score is None:
        score = recovered["score"]
    if city is None:
        city = recovered["city"]
    if field is None:
        field = recovered["field"]
    if profile_subjects is None:
        profile_subjects = recovered["profile_subjects"]

    extracted = {
        "score": score,
        "field": field,
        "city": city,
        "profile_subjects": profile_subjects
    }

    # if score missing -> ask question (no DB)
    if score is None:
        answer = ask_ai(history, extracted, db_results=[])
        history.append({"role": "assistant", "content": answer})
        session_histories[session_id] = history[-MAX_CONTEXT_MESSAGES:]
        return {"reply": answer, "extracted": extracted, "results_count": 0}

    # fetch from DB
    try:
        results = fetch_programs(score, field, city, profile_subjects, limit=10)
    except Exception as e:
        err = f"❌ DB ERROR: {e}"
        history.append({"role": "assistant", "content": err})
        session_histories[session_id] = history[-MAX_CONTEXT_MESSAGES:]
        return {"reply": err, "extracted": extracted, "results_count": 0}

    answer = ask_ai(history, extracted, db_results=results)
    history.append({"role": "assistant", "content": answer})

    # keep memory small
    session_histories[session_id] = history[-MAX_CONTEXT_MESSAGES:]
    return {"reply": answer, "extracted": extracted, "results_count": len(results)}


@app.delete("/reset/{session_id}")
def reset_session(session_id: str):
    session_histories.pop(session_id, None)
    return {"ok": True}
