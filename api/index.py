from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API kulcs és client inicializálása
api_key = os.getenv("NVIDIA_API_KEY")
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

SYSTEM_PROMPT = (
    "Egy professzionális, segítőkész ügyfélszolgálati asszisztens vagy. "
    "Segíts az ügyfélnek időpontot foglalni vagy válaszolj a kérdéseire udvariasan, magyar nyelven."
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not api_key:
        return {"reply": "Hiba: Az NVIDIA_API_KEY hiányzik a Vercel beállításokból!"}

    try:
        # Tisztított üzenetelőzmény összeállítása
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for msg in req.history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_messages.append({
                    "role": str(msg["role"]),
                    "content": str(msg["content"])
                })

        # Hozzáadjuk a felhasználó friss üzenetét
        formatted_messages.append({"role": "user", "content": req.message})

        # AI hívása
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=formatted_messages,
            temperature=0.5,
            max_tokens=1024
        )

        reply_content = response.choices[0].message.content
        
        if reply_content:
            return {"reply": reply_content}
        else:
            return {"reply": "Az AI válasza üres volt. Kérlek próbáld újra!"}

    except Exception as e:
        # Ha bármi hiba történik az NVIDIA hívásban, pontosan kiírja a képernyőre
        return {"reply": f"API Hiba történt: {str(e)}"}
