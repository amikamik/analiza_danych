from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

# Zezwolenie na połączenia (np. z Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/test")
def smoke_test():
    """Nasz endpoint do 'Testu Dymnego'"""
    return {"status": "ok", "message": "Backend na Render działa!"}