## {{APP_NAME}} – Artificial Intelligence Speech-To-Analysis-Translation Engine ({{APP_VERSION}})

**Autor:** pawlict
W razie błędów, problemów technicznych, sugestii ulepszeń lub pomysłów na rozwój aplikacji — napisz do autora: pawlict@proton.me

<p align="center">
  <img src="assets/logo.png" alt="Logo" width="280" />
</p>

---

## Licencja (MIT) i zastrzeżenie (AS IS)

Ten projekt jest udostępniany na licencji **MIT**, która pozwala na używanie, kopiowanie, modyfikowanie, łączenie,
publikowanie, dystrybuowanie, sublicencjonowanie i/lub sprzedaż kopii oprogramowania, pod warunkiem dołączenia informacji
o prawach autorskich oraz treści licencji.

**Wyłączenie gwarancji („AS IS”):**  
Oprogramowanie jest dostarczane *„tak jak jest”*, bez jakiejkolwiek gwarancji — wyraźnej ani dorozumianej — w tym m.in.
bez gwarancji przydatności handlowej, przydatności do określonego celu oraz nienaruszania praw osób trzecich.

**Ograniczenie odpowiedzialności:**  
W żadnym wypadku autor(zy) ani właściciel(e) praw autorskich nie ponoszą odpowiedzialności za roszczenia, szkody
ani inną odpowiedzialność (umowną, deliktową lub inną), wynikającą z używania oprogramowania albo związaną z nim
w jakikolwiek sposób.

---

## Co robi aplikacja

### Transkrypcja audio → tekst (AI)
- **Whisper (openai-whisper)** służy do transkrypcji nagrań na tekst.  
  Modele są **pobierane automatycznie przy pierwszym użyciu** (np. tiny/base/small/medium/large).

### Diaryzacja po głosie (AI)
- **pyannote.audio** służy do rozpoznania „kto mówił kiedy” (speaker diarization) z użyciem pipeline z Hugging Face
  (np. `pyannote/speaker-diarization-community-1`).  
  Może to wymagać:
  - poprawnego **tokenu Hugging Face**,
  - zaakceptowania warunków konkretnego modelu („gated”),
  - przestrzegania licencji/warunków z karty modelu.

### „Diaryzacja” tekstu (bez AI / heurystyka)
- Opcje **Diaryzacji tekstu** (np. naprzemienna / blokowa) **nie używają AI**.
- Działają na gotowym tekście (np. wklejonym lub po transkrypcji Whisper) i przypisują etykiety **SPK1, SPK2, …**
  na podstawie prostych reguł, np.:
  - podział na linie lub zdania,
  - przypisywanie naprzemienne,
  - grupowanie blokami,
  - opcjonalne scalanie krótkich fragmentów.

> To jest formatowanie/porządkowanie tekstu, a nie realne rozpoznanie mówców z audio.

---

## Biblioteki i komponenty zewnętrzne

> Uwaga: część pakietów instaluje się jako zależności pośrednie.
> Lista poniżej skupia się na głównych komponentach używanych przez aplikację.

### GUI
- **PySide6 (Qt for Python)** — interfejs aplikacji (karty, widżety, dialogi, QTextBrowser).  
  Licencja: **LGPL** (Qt for Python).

### Mowa / diaryzacja
- **openai-whisper** — transkrypcja mowy na tekst (Whisper).  
- **pyannote.audio** — pipeline diaryzacji mówców.  
- **huggingface_hub** — pobieranie modeli/pipeline z Hugging Face Hub.

### Rdzeń ML / audio (zwykle jako zależności)
- **torch** (PyTorch) — runtime sieci neuronowych używany przez Whisper / pyannote.
- **torchaudio** — narzędzia audio dla PyTorch (często wymagane przez pyannote).
- **numpy** — obliczenia numeryczne.
- **tqdm** — paski postępu (widać je w logach podczas transkrypcji).
- **soundfile / librosa** — I/O audio i narzędzia (zależnie od środowiska).

### Konwersja audio (narzędzie systemowe)
- **FFmpeg** — konwersja do stabilnego formatu **PCM WAV** (np. 16kHz mono), gdy jest potrzebna.  
  Licencja: zależy od dystrybucji/buildu (warianty LGPL/GPL).

---

## Modele AI (wagi) i warunki użycia

### Wagi modeli Whisper
Wagi modeli Whisper mogą być pobierane automatycznie przy pierwszym użyciu.

**Licencja/warunki wag modelu mogą różnić się od licencji biblioteki Python.**  
Zawsze dokumentuj:
- źródło plików modelu,
- licencję/warunki dotyczące wag.

### Pipeline pyannote (Hugging Face)
Diaryzacja głosu używa repozytorium pipeline na Hugging Face.  
**Użytkownik odpowiada za przestrzeganie licencji/warunków widocznych na karcie konkretnego repozytorium.**


