import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# ── Schema ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    fileContent: str = ""


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="NelsonAI Agent API")
memory = MemorySaver()

# ── CORS ──────────────────────────────────────────────────────────────────────
# Add your exact Vercel URL to this list once you know it.
# Keeping "*" here is fine for initial testing; tighten it before going live.
origins = [
    "*",
    # "https://your-project.vercel.app",   # ← replace with your Vercel URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,        # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM & Tools ───────────────────────────────────────────────────────────────
# Keys are injected as Hugging Face Space Secrets → Environment Variables.
# Never hardcode them here.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

search = TavilySearchResults(k=3)
tools = [search]

# ── Agent ─────────────────────────────────────────────────────────────────────
agent_executor = create_react_agent(llm, tools, checkpointer=memory)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "NelsonAI Backend is running. Visit /docs to explore the API."}


@app.post("/chat")
async def chat(request_data: ChatRequest):
    user_msg = request_data.message
    file_context = request_data.fileContent

    # Each user gets their own memory thread.
    # In production you would pass a real user/session ID from the frontend.
    config = {"configurable": {"thread_id": "user_1"}}

    full_prompt = (
        f"Context from uploaded file:\n{file_context}\n\nUser: {user_msg}"
        if file_context
        else user_msg
    )

    inputs = {"messages": [HumanMessage(content=full_prompt)]}
    result = agent_executor.invoke(inputs, config=config)

    return {"response": result["messages"][-1].content}


# ── Entry point (used by Dockerfile CMD) ──────────────────────────────────────
# Hugging Face Spaces require port 7860.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
