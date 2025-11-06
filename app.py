import pandas as pd
import pingouin as pg  # <-- NOWA BIBLIOTEKA
import itertools        # <-- Do tworzenia par zmiennych
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ydata_profiling import ProfileReport

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === NOWA FUNKCJA DO TESTÓW I TABELI ===
def run_inferential_tests_and_build_table(df: pd.DataFrame) -> str:
    """
    Uruchamia testy statystyczne (na razie t-Testa) dla wszystkich
    pasujących par zmiennych i zwraca tabelę HTML z wynikami.
    """
    results = []
    
    # 1. Znajdź zmienne ciągłe (numeryczne)
    continuous_cols = df.select_dtypes(include='number').columns.tolist()
    
    # 2. Znajdź zmienne kategoryczne, które mają DOKŁADNIE 2 grupy (binarne)
    binary_cols = [col for col in df.columns if 
                   not pd.api.types.is_numeric_dtype(df[col]) and 
                   df[col].nunique() == 2]
    
    # Jeśli nie ma par, zwróć pusty string
    if not binary_cols or not continuous_cols:
        return ""

    # 3. Stwórz wszystkie możliwe pary (ciągła vs binarna)
    for cont_col, bin_col in itertools.product(continuous_cols, binary_cols):
        
        # Pomijaj bezsensowne pary (np. ID vs Płeć)
        if cont_col == bin_col or cont_col == 'ID':
            continue
            
        try:
            # Użyj pingouin do uruchomienia t-Testa
            # pingouin jest mądry - sam obsługuje podział na grupy
            ttest_result = pg.ttest(df[cont_col], df[bin_col], correction=True)
            
            # Wyciągnij kluczowe wartości z ramki danych, którą zwraca pingouin
            p_value = ttest_result['p-val'].iloc[0]
            cohen_d = ttest_result['cohen-d'].iloc[0]
            
            results.append({
                "Zmienna Ciągła": cont_col,
                "Zmienna Grupująca": bin_col,
                "Test": "T-Test (Welch)",
                "p-value": f"{p_value:.4f}", # Zaokrąglij p-value
                "Istotność": "TAK" if p_value < 0.05 else "Nie",
                "d Cohena (Siła Efektu)": f"{cohen_d:.3f}"
            })
            
        except Exception as e:
            # Pomiń błędy (np. gdy dane są złe)
            print(f"Błąd testu {cont_col} vs {bin_col}: {e}")
            pass

    # 4. Zbuduj tabelę HTML z wynikami
    if not results:
        return "<h2>Brak wyników testów statystycznych</h2><p>Nie znaleziono odpowiednich par zmiennych (ciągła vs kategoryczna z 2 grupami) do uruchomienia t-Testów.</p>"

    html = "<h2>Wyniki Testów Istotności (T-Testy)</h2>"
    html += "<p>Porównanie zmiennych ciągłych między dwiema grupami kategorycznymi.</p>"
    html += "<table border='1' style='width:100%; border-collapse: collapse; text-align: left;'><thead><tr>"
    # Nagłówki tabeli
    for key in results[0].keys():
        html += f"<th>{key}</th>"
    html += "</tr></thead><tbody>"
    
    # Wiersze tabeli
    for res in results:
        # Podświetl istotne wiersze
        style = "background-color: #e6ffec;" if res["Istotność"] == "TAK" else ""
        html += f"<tr style='{style}'>"
        for val in res.values():
            html += f"<td>{val}</td>"
        html += "</tr>"
        
    html += "</tbody></table>"
    return html

# === ZAKTUALIZOWANY ENDPOINT GŁÓWNY ===
@app.post("/api/generate-report", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Błąd</h1><p>Plik CSV jest uszkodzony. Błąd: {e}</p>", status_code=400)

    # Krok 1: Wygeneruj raport opisowy (tak jak wcześniej)
    profile = ProfileReport(df, title="Część 1: Automatyczny Raport Opisowy")
    report1_html = profile.to_html()
    
    # Krok 2: Wygeneruj nasz nowy raport wnioskujący (tabela HTML)
    report2_html = run_inferential_tests_and_build_table(df)
    
    # Krok 3: Połącz oba raporty w jeden plik
    final_html = report1_html + "<hr><br>" + report2_html
    
    return HTMLResponse(content=final_html)

@app.get("/api/test")
def smoke_test():
    return {"status": "ok", "message": "Backend na Render działa!"}