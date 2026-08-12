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

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

SYSTEM_PROMPT = (
    "Egy professzionális ügyfélszolgálati asszisztens vagy. "
    "Segíts az ügyfélnek időpontot foglalni vagy válaszolj a kérdéseire."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Lekéri a szabad időpontokat.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Lefoglal egy időpontot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "phone": {"type": "string"},
                    "service_type": {"type": "string"},
                    "slot": {"type": "string"}
                },
                "required": ["name", "phone", "service_type", "slot"]
            }
        }
    }
]

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.history + [{"role": "user", "content": req.message}]
        
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        response_msg = response.choices[0].message

        if response_msg.tool_calls:
            # Llama válaszának rögzítése szótár formátumban (kivédi a ChatCompletionMessage hibát)
            messages.append({
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [tc.model_dump() for tc in response_msg.tool_calls]
            })

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name == "get_available_slots":
                    res = "Szabad időpontok: Holnap 10:00, Holnap 14:00, Péntek 11:30"
                elif func_name == "book_appointment":
                    res = f"Sikeres foglalás! Név: {args.get('name')}, Időpont: {args.get('slot')}"
                else:
                    res = "Művelet elvégezve."

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})

            final_res = client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=messages
            )
            return {"reply": final_res.choices[0].message.content}
        
        return {"reply": response_msg.content or "Sajnálom, nem tudtam feldolgozni a kérést."}

    except Exception as e:
        return {"reply": f"Szerver hiba történt: {str(e)}"}
