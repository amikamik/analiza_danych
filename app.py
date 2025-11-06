import pandas as pd
import pingouin as pg
import itertools
import io
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

# === NASZA NOWA, "AKADEMICKA" LOGIKA TESTOWANIA ===
def run_academic_tests_and_build_table(df: pd.DataFrame) -> str:
    """
    Uruchamia testy statystyczne zgodnie z rygorystycznym podejściem
    akademickim, sprawdzając założenia przed wykonaniem testu.
    """
    results = []
    
    # 1. Zidentyfikuj typy zmiennych
    continuous_cols = df.select_dtypes(include='number').columns.tolist()
    all_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_cols = [col for col in all_categorical_cols if df[col].nunique() == 2]

    # --- SCENARIUSZ 1: Ciągła vs. Binarna (Test T-Studenta) ---
    for cont_col, bin_col in itertools.product(continuous_cols, binary_cols):
        if cont_col.lower() == 'id' or bin_col.lower() == 'id':
            continue
            
        try:
            # 1. Sprawdź założenie o normalności rozkładu w obu grupach
            # Używamy testu Shapiro-Wilka. p > 0.05 = rozkład normalny.
            normality_result = pg.normality(data=df, dv=cont_col, group=bin_col)
            is_normal = normality_result['p-val'].min() > 0.05

            if not is_normal:
                results.append({
                    "Zmienne": f"{cont_col} vs. {bin_col}",
                    "Typ Analizy": "Ciągła vs. Binarna",
                    "Użyty Test": "Test T-Studenta",
                    "Status": "Nie wykonano",
                    "p-value": "N/A",
                    "Siła Efektu": "N/A",
                    "Uwagi": "Założenie o normalności rozkładu nie zostało spełnione."
                })
                continue # Przejdź do następnej pary

            # 2. Skoro jest normalność, sprawdź równość wariancji (Test Levene'a)
            levene_result = pg.homoscedasticity(data=df, dv=cont_col, group=bin_col)
            is_homoscedastic = levene_result['p-val'].iloc[0] > 0.05
            correction = not is_homoscedastic # Użyj korekty Welcha, jeśli wariancje NIE są równe
            test_name = "Test T-Studenta" if is_homoscedastic else "Test T (Welch)"

            # 3. Założenia spełnione, wykonaj test
            test_result = pg.ttest(data=df, dv=cont_col, between=bin_col, correction=correction)
            p_value = test_result['p-val'].iloc[0]
            cohen_d = test_result['cohen-d'].iloc[0]

            results.append({
                "Zmienne": f"{cont_col} vs. {bin_col}",
                "Typ Analizy": "Ciągła vs. Binarna",
                "Użyty Test": test_name,
                "Status": "Wykonano",
                "p-value": f"{p_value:.4f}",
                "Siła Efektu": f"d Cohena = {cohen_d:.3f}",
                "Uwagi": f"Założenia spełnione (Normalność: Tak, Równość Wariancji: {'Tak' if is_homoscedastic else 'Nie'})."
            })
            
        except Exception as e:
            results.append({"Zmienne": f"{cont_col} vs. {bin_col}", "Typ Analizy": "Ciągła vs. Binarna", "Status": f"Błąd: {e}", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "N/A"})
            pass

    # --- SCENARIUSZ 2: Ciągła vs. Ciągła (Regresja Liniowa) ---
    for col1, col2 in itertools.combinations(continuous_cols, 2):
        if col1.lower() == 'id' or col2.lower() == 'id':
            continue
        try:
            # Uruchom regresję liniową (Y ~ X)
            lm_result = pg.linear_regression(df[col1].dropna(), df[col2].dropna())
            # Weź p-value dla predyktora (col2)
            p_value = lm_result.loc[1, 'p-val'] # [1] to wiersz dla zmiennej, [0] to intercept
            r_squared = lm_result['r2'].iloc[0]
            
            results.append({
                "Zmienne": f"{col1} vs. {col2}",
                "Typ Analizy": "Ciągła vs. Ciągła",
                "Użyty Test": "Regresja Liniowa",
                "Status": "Wykonano",
                "p-value": f"{p_value:.4f}",
                "Siła Efektu": f"R-kwadrat = {r_squared:.3f}",
                "Uwagi": "Założenia regresji nie zostały sprawdzone (MVP)."
            })
        except Exception as e:
            results.append({"Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Ciągła vs. Ciągła", "Status": f"Błąd: {e}", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "N/A"})
            pass

    # --- SCENARIUSZ 3: Kategoryczna vs. Kategoryczna (Chi-kwadrat) ---
    for col1, col2 in itertools.combinations(all_categorical_cols, 2):
        try:
            chi2_result = pg.chi2_independence(data=df, x=col1, y=col2)
            p_value = chi2_result[2][1]
            cramer_v = chi2_result[2][2]
            
            # Sprawdź założenie o liczebnościach oczekiwanych
            expected = chi2_result[1]
            assumption_met = expected.min().min() >= 5
            
            if assumption_met:
                results.append({
                    "Zmienne": f"{col1} vs. {col2}",
                    "Typ Analizy": "Kategoryczna vs. Kategoryczna",
                    "Użyty Test": "Test Chi-kwadrat",
                    "Status": "Wykonano",
                    "p-value": f"{p_value:.4f}",
                    "Siła Efektu": f"V Craméra = {cramer_v:.3f}",
                    "Uwagi": "Założenia spełnione (liczebności oczekiwane > 5)."
                })
            else:
                results.append({
                    "Zmienne": f"{col1} vs. {col2}",
                    "Typ Analizy": "Kategoryczna vs. Kategoryczna",
                    "Użyty Test": "Test Chi-kwadrat",
                    "Status": "Nie wykonano",
                    "p-value": "N/A",
                    "Siła Efektu": "N/A",
                    "Uwagi": "Założenie o liczebnościach oczekiwanych (> 5) nie zostało spełnione."
                })
        except Exception as e:
            results.append({"Zmienne": f"{col1} vs. {col2}", "Typ Analizy": "Kategoryczna vs. Kategoryczna", "Status": f"Błąd: {e}", "p-value": "N/A", "Siła Efektu": "N/A", "Uwagi": "N/A"})
            pass

    # 4. Zbuduj tabelę HTML z wynikami
    if not results:
        return "<h2>Brak wyników testów statystycznych</h2><p>Nie znaleziono odpowiednich par zmiennych do analizy.</p>"

    # Sortuj wyniki, aby istotne były na górze
    results.sort(key=lambda x: (x["Status"], x["p-value"]))

    html = "<h2>Część 2: Wyniki Testów Wnioskujących (Istotności)</h2>"
    html += "<p>Automatyczna analiza zależności między zmiennymi. Istotne wyniki (p<0.05) są podświetlone.</p>"
    html += "<table border='1' style='width:100%; border-collapse: collapse; text-align: left; font-size: 14px;'><thead><tr style='background-color: #f0f0f0;'>"
    
    # Nagłówki tabeli
    headers = ["Zmienne", "Typ Analizy", "Użyty Test", "Status", "p-value", "Siła Efektu", "Uwagi"]
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    
    # Wiersze tabeli
    for res in results:
        is_significant = res["Status"] == "Wykonano" and float(res["p-value"]) < 0.05
        style = "background-color: #e6ffec; font-weight: bold;" if is_significant else ""
        if res["Status"] == "Nie wykonano":
            style = "background-color: #fff0f0;" # Lekki czerwony dla niewykonanych

        html += f"<tr style='{style}'>"
        for key in headers:
            html += f"<td>{res.get(key, 'N/A')}</td>" # Użyj .get() dla bezpieczeństwa
        html += "</tr>"
        
    html += "</tbody></table>"
    return html

# === ZAKTUALIZOWANY ENDPOINT GŁÓWNY ===
@app.post("/api/generate-report", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    
    file_content = await file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        return HTMLResponse(content=f"<h1>Błąd</h1><p>Plik CSV jest uszkodzony. Błąd: {e}</p>", status_code=400)

    # Krok 1: Wygeneruj raport opisowy (minimalny, aby oszczędzać RAM)
    # Wyłączamy drogie obliczenia (korelacje, chi2), bo robimy je sami!
    profile = ProfileReport(df, 
                            title="Część 1: Automatyczny Raport Opisowy (Podstawowy)", 
                            minimal=True, 
                            correlations=None, 
                            interactions=None, 
                            missing_diagrams=None)
    report1_html = profile.to_html()
    
    # Krok 2: Wygeneruj nasz nowy, inteligentny raport wnioskujący
    report2_html = run_academic_tests_and_build_table(df)
    
    # Krok 3: Połącz oba raporty w jeden plik
    final_html = report1_html + "<br><hr style='border: 2px solid #007bff;'>" + report2_html
    
    return HTMLResponse(content=final_html)

@app.get("/api/test")
def smoke_test():
    return {"status": "ok", "message": "Backend na Render działa!"}