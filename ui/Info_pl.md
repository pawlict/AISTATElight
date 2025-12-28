## {{APP_NAME}} – Artificial Intelligence Speech-To-Analysis-Translation Engine ({{APP_VERSION}})

**Autor:** pawlict
---
(Kontakt)
Jeśli napotkasz błędy, problemy techniczne, masz sugestie dotyczące ulepszeń lub pomysły na nowe funkcje, skontaktuj się z autorem pod adresem: pawlict@proton.me
<p align="center">
  <img src="assets/logo.png" alt="Logo" width="280" />
</p>

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

---
---

## Licencja i zastrzeżenia

Ten projekt jest udostępniany na licencji:  
**AISTATElight GitHub-Compatible No-Resale License v1.2 (Source-Available)**

### ✅ Uprawnienia
Użytkownik może:
- korzystać z oprogramowania w celach **prywatnych, edukacyjnych, badawczych** oraz **komercyjnych na potrzeby wewnętrzne**,
- **modyfikować** kod na własne potrzeby (wewnętrznie).

### ❌ Ograniczenia
Bez uprzedniej **pisemnej zgody autora** użytkownik **nie może**:
- odsprzedawać ani prowadzić **komercyjnej redystrybucji** oprogramowania (kod źródłowy lub wersje binarne),
- dystrybuować **binarek/instalatorów/kontenerów** ani innych paczek wdrożeniowych,
- publikować zmodyfikowanych wersji poza mechanizmami platformy GitHub (fork/view), w zakresie w jakim GitHub dopuszcza to regulaminem.

### © Wzmianka o autorze (wymagana)
Wymagane jest zachowanie informacji o autorze (**pawlict**) oraz treści licencji.
W przypadku użycia komercyjnego w organizacji atrybucja powinna pozostać widoczna w sekcji **About/Info** (lub równoważnej dokumentacji).

### „AS IS” — wyłączenie gwarancji
Oprogramowanie jest dostarczane **„TAK JAK JEST”**, bez jakichkolwiek gwarancji (wyraźnych ani dorozumianych),
w tym m.in. bez gwarancji przydatności handlowej, przydatności do określonego celu oraz nienaruszania praw osób trzecich.

### Ograniczenie odpowiedzialności
W żadnym wypadku autor ani właściciel praw autorskich nie ponosi odpowiedzialności za roszczenia, szkody ani inną odpowiedzialność
wynikającą z używania oprogramowania lub związaną z nim w jakikolwiek sposób.

### Licencje zależności (Third-party)
To oprogramowanie korzysta z bibliotek i narzędzi stron trzecich (np. Whisper, pyannote.audio, PySide6/Qt, FFmpeg),
które mają własne licencje. Szczegóły: **THIRD_PARTY_NOTICES.md**.

---


