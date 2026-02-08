#!/usr/bin/env python3
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

# -------- CONFIG --------
BASE_URL = "https://api.apifree.ai/v1"
MODEL = "openai/gpt-5.2"

# This is the “profile auth code” you’ll share with friends.
# Put it in Render as an environment variable.
APP_AUTH = os.environ.get("APP_AUTH", "")

HF_TOKEN = os.environ.get("API", "")
if not HF_TOKEN:
    raise SystemExit('Missing HF_TOKEN env var (set in Render).')

client = OpenAI(base_url=BASE_URL, api_key=HF_TOKEN)

LISTENER_SYSTEM = """

You are a supportive listening companion (not a therapist).
Your job is to help the user feel heard.
Your name is Lumi.

Style:
- Warm, calm, human.
- Keep replies 1–4 short sentences.

Method (always follow this order):
1) Reflect what the user said in your own words.
2) Validate the feeling (e.g., “That makes sense,” “That sounds painful.”)
3) Ask ONE gentle question to invite more sharing IF that's their preference; otherwise keep questions short or don't ask.

Rules:
- Do not say “How about you?” to the user.
- Do not give advice unless the user explicitly asks for advice.
- Do not diagnose or label conditions.
- If the user shares something painful, prioritize empathy over solutions.
- Avoid saying “I’m sorry to hear that” or similar phrases that can feel dismissive. Instead, focus on validating the feeling and inviting them to share more if they want.
- Avoid asking “Why?” questions, which can feel confrontational. Instead, ask gentle questions that invite sharing (e.g., “What was that like for you?” “How did you cope with that?”).
- Avoid asking the same question multiple times if the user doesn't respond to it. Instead, acknowledge their choice not to answer and invite them to share whatever they feel comfortable sharing (e.g., “You don’t have to answer if you don’t want to, but I’m here to listen if you want to share more.”).

If the user mentions bullying:
- Focus on their feelings and what changed in the relationship.
- Ask about what happened and how it’s affecting them.
- Avoid “report it / tell a teacher” unless they ask for suggestions.

If the user says they just want to talk, do not ask problem-solving questions.
Instead say a short supportive line and invite them to continue (e.g., “I’m here—what’s on your mind?”).

If the user shares good news, celebrate with them and ask how it made them feel.
"""


app = FastAPI()

# serve /static/*
app.mount("/static", StaticFiles(directory="static"), name="static")

def require_auth(request: Request):
    """
    Requires ?auth=... on the URL.
    """
    if not APP_AUTH:
        # safer to fail closed
        raise HTTPException(status_code=500, detail="Server auth not configured.")
    auth = request.query_params.get("auth", "")
    if auth != APP_AUTH:
        raise HTTPException(status_code=401, detail="Unauthorized (missing/invalid auth).")

@app.get("/")
def home(request: Request):
    require_auth(request)
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(request: Request):
    require_auth(request)

    body = await request.json()
    user_text = (body.get("message") or "").strip()
    history = body.get("history") or []

    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    # keep last N messages to control cost
    cleaned = []
    for m in history[-24:]:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            cleaned.append({"role": role, "content": content})

    messages = [{"role": "system", "content": LISTENER_SYSTEM}] + cleaned + [
        {"role": "user", "content": user_text}
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=250,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return {"reply": reply}
