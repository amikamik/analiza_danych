import pandas as pd
import pingouin as pg
import itertools
import io
# === ZMIANA 1: Musimy zaimportować HTTPException ===
from fastapi import FastAPI, File, UploadFile, HTTPException 
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ydata_profiling import ProfileReport

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === KROK 1: ENDPOINT PODGLĄDU (Z POPRAWKĄ BŁĘDU) ===
@app.post("/api/parse-preview", response_class=JSONResponse)
async def parse_preview(file: UploadFile = File(...)):
    """
    Wczytuje tylko pierwsze 10 wierszy pliku CSV, aby pobrać nazwy kolumn
    i dane do podglądu. Zwraca je jako JSON.
    """
    try:
        # Próba odczytania CSV z różnymi popularnymi kodowaniami
        # To powinno naprawić błąd odczytu
        try:
            df_preview = pd.read_csv(file.file, nrows=10, encoding='utf-8')
        except UnicodeDecodeError:
            # Jeśli UTF-8 zawiedzie, spróbuj popularnego w Windows
            file.file.seek(0) # Wróć na początek pliku
            df_preview = pd.read_csv(file.file, nrows=10, encoding='latin1')

        columns = df_preview.columns.tolist()
        preview_data = df_preview.values.tolist()
        
        return {
            "columns": columns,
            "preview_data": preview_data
        }
        
    except Exception as e:
        # === ZMIANA 2: Używamy HTTPException zamiast JSONResponse ===
        # To gwarantuje, że nagłówki CORS zostaną poprawnie dołączone
        raise HTTPException(
            status_code=400,
            detail=f"Nie udało się przetworzyć pliku CSV. Upewnij się, że ma poprawne kodowanie (np. UTF-8). Błąd: {e}"
        )

# === PONIŻEJ RESZTA KODU (BEZ ZMIAN) ===

def run_academic_tests_and_build_table(df: pd.DataFrame) -> str:
    # ... (cała ta duża, "akademicka" funkcja zostaje tutaj bez zmian) ...
    results = []
    
    # 1. Zidentyfikuj typy zmiennych
    all_numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()

    continuous_cols = [col for col in all_numeric_cols if df[col].nunique() > 2 and col.lower() != 'id']
    binary_numeric = [col for col in all_numeric_cols if df[col].nunique() == 2]
    binary_categorical = [col for col in all_categorical_cols if df[col].nunique() == 2]
    binary_cols = binary_categorical + binary_numeric + bool_cols
    categorical_cols = all_categorical_cols + binary_numeric + bool_cols
    categorical_cols = list(set(categorical_cols)) 

    # --- SCENARIUSZ 1: Ciągła vs. Binarna ---
    for cont_col, bin_col in itertools.product(continuous_cols, binary_cols):
        try:
            normality_result = pg.normality(data=df, dv=cont_col, group=bin_col)
            is_normal = normality_result['p-val'].min() > 0.05
            if not is_normal:
                results.append({ "Zmienne": f"{cont_col} vs. {bin_col}", "Typ Analizy": "Ciągła vs. Binarna", "Użyty Test": "Test T-Studenta", "Status": "Nie wykonano", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "Założenie o normalności rozkładu nie zostało spełnione." })
                continue 
            levene_result = pg.homoscedasticity(data=df, dv=cont_col, group=bin_col)
            is_homoscedastic = levene_result['p-val'].iloc[0] > 0.05
            correction = not is_homoscedastic 
            test_name = "Test T-Studenta" if is_homoscedastic else "Test T (Welch)"
            test_result = pg.ttest(data=df, dv=cont_col, between=bin_col, correction=correction)
            p_value = test_result['p-val'].iloc[0]
            cohen_d = test_result['cohen-d'].iloc[0]
            results.append({ "Zmienne": f"{cont_col} vs. {bin_col}", "Typ Analizy": "Ciągła vs. Binarna", "Użyty Test": test_name, "Status": "Wykonano", "p-value": f"{p_value:.4f}", "Siła Efektu": f"d Cohena = {cohen_d:.3f}", "Uwagi": f"Założenia spełnione (Normalność: Tak, Równość Wariancji: {'Tak' if is_homoscedastic else 'Nie'})." })
        except Exception as e:
            results.append({"Zmienne": f"{cont_col} vs. {bin_col}", "Typ Analizy": "Ciągła vs. Binarna", "Status": f"Błąd: {e}", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "N/A"})

    # --- SCENARIUSZ 2: Ciągła vs. Ciągła ---
    for col1, col2 in itertools.combinations(continuous_cols, 2):
        try:
            lm_result = pg.linear_regression(df[col1].dropna(), df[col2].dropna())
            p_value = lm_result.loc[1, 'p-val'] 
            r_squared = lm_result['r2'].iloc[0]
            results.append({ "Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Ciągła vs. Ciągła", "Użyty Test": "Regresja Liniowa", "Status": "Wykonano", "p-value": f"{p_value:.4f}", "Siła Efektu": f"R-kwadrat = {r_squared:.3f}", "Uwagi": "Założenia regresji nie zostały sprawdzone (MVP)." })
        except Exception as e:
            results.append({"Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Ciągła vs. Ciągła", "Status": f"Błąd: {e}", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "N/A"})

    # --- SCENARIUSZ 3: Kategoryczna vs. Kategoryczna ---
    for col1, col2 in itertools.combinations(categorical_cols, 2):
        try:
            chi2_result = pg.chi2_independence(data=df, x=col1, y=col2)
            stats_df = chi2_result[2]
            p_value = stats_df.loc[stats_df['test'] == 'pearson', 'p-val'].iloc[0]
            cramer_v = stats_df.loc[stats_df['test'] == 'pearson', 'cramer'].iloc[0]
            expected = chi2_result[1]
            assumption_met = expected.min().min() >= 5
            if assumption_met:
                results.append({ "Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Kategoryczna vs. Kategoryczna", "Użyty Test": "Test Chi-kwadrat", "Status": "Wykonano", "p-value": f"{p_value:.4f}", "Siła Efektu": f"V Craméra = {cramer_v:.3f}", "Uwagi": "Założenia spełnione (liczebności oczekiwane > 5)." })
            else:
                results.append({ "Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Kategoryczna vs. Kategoryczna", "Użyty Test": "Test Chi-kwadrat", "Status": "Nie wykonano", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "Założenie o liczebnościach oczekiwanych (> 5) nie zostało spełnione." })
        except Exception as e:
            results.append({"Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Kategoryczna vs. Kategoryczna", "Status": f"Błąd: {e}", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "N/A"})

    # --- Budowanie tabeli HTML (bez zmian) ---
    if not results:
        return "<h2>Brak wyników testów statystycznych</h2><p>Nie znaleziono odpowiednich par zmiennych do analizy.</p>"
    results.sort(key=lambda x: (x["Status"], x["p-value"]))
    html = "<h2>Część 2: Wyniki Testów Wnioskujących (Istotności)</h2>"
    html += "<p>Automatyczna analiza zależności między zmiennymi. Istotne wyniki (p<0.05) są podświetlone.</p>"
    html += "<table border='1' style='width:100%; border-collapse: collapse; text-align: left; font-size: 14px;'><thead><tr style='background-color: #f0f0f0;'>"
    headers = ["Zmienne", "Typ Analizy", "Użyty Test", "Status", "p-value", "Siła Efektu", "Uwagi"]
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    for res in results:
        is_significant = res["Status"] == "Wykonano" and float(res["p-value"]) < 0.05
        style = "background-color: #e6ffec; font-weight: bold;" if is_significant else ""
        if res["Status"] == "Nie wykonano":
            style = "background-color: #fff0f0;"
        html += f"<tr style='{style}'>"
        for key in headers:
            html += f"<td>{res.get(key, 'N/A')}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# === GŁÓWNY ENDPOINT (NA RAZIE BEZ ZMIAN) ===
@app.post("/api/generate-report", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    
    file_content = await file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        return HTMLResponse(content=f"<h1>Błąd</h1><p>Plik CSV jest uszkodzony. Błąd: {e}</p>", status_code=400)

    # (Używam Twojej wersji z `minimal=True`, aby zapewnić stabilność)
    profile = ProfileReport(df, 
                            title="Część 1: Automatyczny Raport Opisowy (Podstawowy)", 
                            minimal=True, 
                            correlations=None, 
                            interactions=None, 
                            missing_diagrams=None)
    report1_html = profile.to_html()
    
    report2_html = run_academic_tests_and_build_table(df) 
    
    final_html = report1_html + "<br><hr style='border: 2px solid #007bff;'>" + report2_html
    
    return HTMLResponse(content=final_html)

@app.get("/api/test")
def smoke_test():
    return {"status": "ok", "message": "Backend na Render działa!"}