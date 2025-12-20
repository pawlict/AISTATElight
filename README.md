# AISTATE Light Beta
![Version](https://img.shields.io/badge/version-v1-blue)
![Python](https://img.shields.io/badge/python-3.8+-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

**AISTATE Light Beta** to narzędzie, służące do  transkrypcji i diaryzacji.  

---

## ✨ Główne funkcjonalności

### 1) Diarizacja tekstu (na bazie transkryptu)
Dostępne metody:
- **Szybka – naprzemienna**: oznacza linie jako `[SPK1]`, `[SPK2]`, … (w kółko)
- **Embeddings (liczba mówców)**: embeddings + KMeans dla zadanej liczby mówców
- **Embeddings (auto liczba mówców)**: dobór liczby klastrów przez silhouette score (2..max)

### 2) Audio → transkrypcja (Whisper)
- wybór modelu: `tiny/base/small/medium/large`
- język: np. `pl` (lub puste = auto)
- wynik trafia do lewego panelu jako transkrypt z timestampami

### 3) Audio → transkrypcja + diarizacja po głosie (Whisper + pyannote)
- Whisper robi segmenty czasowe (tekst)
- pyannote robi segmenty mówców (głos)
- aplikacja łączy je po **nakładaniu się w czasie** i generuje wynik w prawym panelu:  
  `"[SPK1][00:00:05–00:00:10] ..."`

---

## Wymagania

- Python: zalecane **3.10–3.12** (3.13 może działać zależnie od pakietów)
- Systemowy **ffmpeg** (wymagany dla audio)
- Biblioteki Python: patrz `requirements.txt`
- Dla diarizacji pyannote: konto HF + token

---
## Hugging Face Token (pyannote)
- Diarizacja głosowa wymaga tokena HF. Wklej token w zakładce Ustawienia → zapisz (aplikacja zapisze do ~/.pyannote_hf_token)
---

## 🚀 Instalacja Linuks

### 1 Aktualizacja systemu
```bash
sudo apt-get update -y
```
### 2 Instalacja pakietów
```bash
sudo apt install -y ffmpeg
sudo apt install python3-tk
```
---
### 3 Instalacja programu
```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/pawlict/AISTATElight.git
cd AISTATElight

python3 -m venv .AISTATElight
source .AISTATElight/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```
---
### 4 Uruchomienie programu
```bash
python3 AISTATElight.py
```
---
### Troubleshooting
## “Unable to locate package telegram-desktop” / brak pakietów w systemie
- To dotyczy APT — tutaj potrzebujesz ffmpeg i Pythona w venv. Upewnij się, że instalujesz pipem w venv.
## Brak diarizacji po głosie
- Sprawdź czy pyannote.audio jest zainstalowane, sprawdź token HF (Ustawienia).
- Czasem model na HF wymaga akceptacji warunków na stronie repozytorium modelu.
## ffmpeg error while converting audio
- Sprawdź czy ffmpeg działa w terminalu: ffmpeg -version
- Spróbuj inne wejściowe audio (czasem pliki mają uszkodzone metadane)
