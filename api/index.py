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
    "Egy professzionális, udvarias ügyfélszolgálati asszisztens vagy.\n\n"
    "SZIGORÚ IDŐPONTFOGLALÁSI SZABÁLYOK:\n"
    "1. Ha a felhasználó időpontot szeretne foglalni, ELŐSZÖR hívd meg a `get_available_slots` funkciót!\n"
    "2. A foglaláshoz KÖTELEZŐ MIND A 4 ADAT MEGLÉTE:\n"
    "   - Név (name)\n"
    "   - Telefonszám (phone)\n"
    "   - Kért szolgáltatás (service_type)\n"
    "   - Kiválasztott időpont (slot)\n"
    "3. HA BÁRMELYIK ADAT HIÁNYZIK A NÉGYBŐL, NE HÍVD MEG a `book_appointment` funkciót! "
    "Ehelyett kérdezd meg a hiányzó adatokat az ügyféltől (pl. 'Kérem adja meg a nevét és telefonszámát is!').\n"
    "4. Ha az ügyfél élő embert/ügyintézőt kér, hívd meg a `transfer_to_human` funkciót!"
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
            "description": "Lefoglal egy időpontot az ügyfélnek.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Az ügyfél teljes neve"},
                    "phone": {"type": "string", "description": "Az ügyfél telefonszáma"},
                    "service_type": {"type": "string", "description": "Kért szolgáltatás"},
                    "slot": {"type": "string", "description": "A kiválasztott időpont"}
                },
                "required": ["name", "phone", "service_type", "slot"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Átkapcsolja az ügyfelet egy élő emberi ügyintézőhöz.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Átirányítás oka"}
                },
                "required": ["reason"]
            }
        }
    }
]

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.history + [{"role": "user", "content": req.message}]
    
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    response_msg = response.choices[0].message

    if response_msg.tool_calls:
        messages.append(response_msg.model_dump())
        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "get_available_slots":
                res = "Szabad időpontjaink: 2026-08-12 10:30, 2026-08-13 11:00, 2026-08-13 15:00."
            elif func_name == "book_appointment":
                res = f"Sikeres foglalás! Név: {args['name']}, Tel: {args['phone']}, Szolgáltatás: {args['service_type']}, Időpont: {args['slot']}."
            elif func_name == "transfer_to_human":
                res = "Az átirányítási kérelmet rögzítettük. Egy munkatársunk hamarosan átveszi a beszélgetést!"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res})

        final_res = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages
        )
        return {"reply": final_res.choices[0].message.content}
    
    return {"reply": response_msg.content}