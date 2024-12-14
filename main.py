from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from utils import cal_len
from utils import wiki_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_input: str

@app.post("/process-data/")
async def process_data(data: UserInput):
    input_length = cal_len(data.user_input)
    agent_response = wiki_agent(data.user_input)
    
    if data.user_input:
        print(f"User Inputs: {data.user_input}")
    
    return {
        "message": "Data received successfully!",
        "input_length": input_length,
        "agent's response": agent_response.get("agent's response"),
        "tool_response": agent_response.get("tool_response"),
        "raw_messages": agent_response.get("raw_messages")
    }

@app.get("/")
async def read_root():
    return {"message": "Hello World!!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)