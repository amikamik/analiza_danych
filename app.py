from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse # Ważne: Zwracamy HTML
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from ydata_profiling import ProfileReport # To jest nasz generator

app = FastAPI()

# Zezwolenie Vercelowi na komunikację z Renderem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint testowy (zostawiamy go)
@app.get("/api/test")
def smoke_test():
    return {"status": "ok", "message": "Backend na Render działa!"}

# --- NASZ GŁÓWNY ENDPOINT DO RAPORTÓW ---
@app.post("/api/generate-report", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    
    # Krok 1: Wczytaj plik CSV używając Pandas
    try:
        # file.file to obiekt pliku wysłany przez użytkownika
        df = pd.read_csv(file.file)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Błąd</h1><p>Plik CSV jest uszkodzony lub ma zły format. Błąd: {e}</p>", status_code=400)

    # Krok 2: Wygeneruj raport "profilujący" z danych
    profile = ProfileReport(df, title="Automatyczny Raport Statystyczny")
    
    # Krok 3: Skonwertuj raport do pojedynczego stringa HTML
    html_report = profile.to_html()
    
    # Krok 4: Zwróć gotowy HTML jako odpowiedź
    return HTMLResponse(content=html_report)