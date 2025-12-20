import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import tempfile
import subprocess
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
import datetime
from typing import List, Optional

# Wersje pakietów (jeśli dostępne)
try:
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except Exception:
    pkg_version = None
    PackageNotFoundError = Exception


# ================== INFO / METADATA ==================

APP_NAME = "Artificial Intelligence Speech-To-Analysis-Translation Engine Light"
APP_VERSION = "Beta 2"
APP_AUTHOR = "pawlict"
APP_LICENSE = "MIT"
APP_LICENSE_ASIS = 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.'

# Informacje o AI/modelach używanych w aplikacji
AI_AND_MODELS_INFO = [
    {
        "name": "OpenAI Whisper (openai-whisper)",
        "purpose": "ASR / transkrypcja audio",
        "license": "MIT",
        "ref": "https://github.com/openai/whisper",
    },
    {
        "name": "pyannote speaker diarization pipeline (pyannote/speaker-diarization-3.1)",
        "purpose": "Diarizacja po barwie głosu",
        "license": "MIT",
        "ref": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    },
    {
        "name": "SentenceTransformers model: sentence-transformers/all-MiniLM-L6-v2",
        "purpose": "Embeddings do diarizacji tekstu (klasteryzacja wypowiedzi)",
        "license": "Apache-2.0",
        "ref": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    },
]

# Użyte biblioteki (oraz licencje – ogólne, wg projektów upstream)
LIBS_INFO = [
    {"name": "Python (stdlib: os, tempfile, subprocess, shutil, tkinter, typing)", "license": "PSF / stdlib"},
    {"name": "openai-whisper", "license": "MIT"},
    {"name": "pyannote.audio", "license": "MIT"},
    {"name": "torch (PyTorch)", "license": "BSD-3-Clause"},
    {"name": "sentence-transformers", "license": "Apache-2.0"},
    {"name": "scikit-learn", "license": "BSD-3-Clause"},
    {"name": "ffmpeg (zewnętrzny binarny)", "license": "LGPL-2.1+ (z opcjonalnymi komponentami GPL)"},
]


def get_installed_version_any(names: List[str]) -> str:
    """Zwraca wersję pierwszego znalezionego pakietu z listy nazw (pip metadata)."""
    if pkg_version is None:
        return "nieznana (brak importlib.metadata)"

    for n in names:
        try:
            return pkg_version(n)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return "niezainstalowane"


def build_info_text() -> str:
    """Składa tekst do zakładki Info."""
    # Uwaga: nazwy pakietów w metadata czasem różnią się od modułów
    versions = {
        "openai-whisper": get_installed_version_any(["openai-whisper", "whisper"]),
        "pyannote.audio": get_installed_version_any(["pyannote.audio", "pyannote-audio"]),
        "torch": get_installed_version_any(["torch"]),
        "sentence-transformers": get_installed_version_any(["sentence-transformers", "sentence_transformers"]),
        "scikit-learn": get_installed_version_any(["scikit-learn", "sklearn"]),
    }

    lines = []
    lines.append(f"{APP_NAME}")
    lines.append("")
    lines.append(f"Wersja: {APP_VERSION}")
    lines.append(f"Autor: {APP_AUTHOR}")
    lines.append("")
    lines.append("Licencja aplikacji:")
    lines.append(f"- {APP_LICENSE}")
    lines.append(f"- {APP_LICENSE_ASIS}")
    lines.append("")
    lines.append("Użyte biblioteki (z wersjami jeśli dostępne):")
    lines.append("- Python (tkinter/ttk i standard library)")

    # Wypisz opcjonalne biblioteki z wersjami
    lines.append(f"- openai-whisper: {versions['openai-whisper']}")
    lines.append(f"- pyannote.audio: {versions['pyannote.audio']}")
    lines.append(f"- torch: {versions['torch']}")
    lines.append(f"- sentence-transformers: {versions['sentence-transformers']}")
    lines.append(f"- scikit-learn: {versions['scikit-learn']}")
    lines.append("- ffmpeg: wymagany w systemie (zewnętrzny program)")

    lines.append("")
    lines.append("Licencje zależności (ogólne):")
    for item in LIBS_INFO:
        lines.append(f"- {item['name']} — {item['license']}")

    lines.append("")
    lines.append("Wykorzystane AI / modele oraz licencje:")
    for m in AI_AND_MODELS_INFO:
        lines.append(f"- {m['name']}")
        lines.append(f"  • Zastosowanie: {m['purpose']}")
        lines.append(f"  • Licencja: {m['license']}")
        lines.append(f"  • Referencja: {m['ref']}")

    lines.append("")
    lines.append("Uwagi techniczne:")
    lines.append("- Aplikacja ustawia TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, aby uniknąć problemów ładowania checkpointów pyannote na PyTorch 2.6+.")
    lines.append('- Pamiętaj: niektóre modele na Hugging Face mogą mieć dodatkowe warunki dostępu (np. akceptacja warunków repozytorium).')

    return "\n".join(lines)


# ====== MODELE TEKSTOWE (EMBEDDINGS) ======
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_NLP_LIBS = True
except ImportError:
    SentenceTransformer = None
    KMeans = None
    silhouette_score = None
    HAS_NLP_LIBS = False

# ====== WHISPER DO AUDIO ======
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    whisper = None
    HAS_WHISPER = False

# ====== PYANNOTE DO DIARYZACJI PO GŁOSIE ======
try:
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except ImportError:
    Pipeline = None
    HAS_PYANNOTE = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".pyannote_hf_token")


# ========= FUNKCJE DIARYZACJI TEKSTU =========

def diarize_alternating(lines: List[str], num_speakers: int) -> List[str]:
    """Najprostsza metoda: SPK1, SPK2, ..., SPK{n}, potem znowu od SPK1."""
    result = []
    speaker_idx = 0

    for line in lines:
        raw = line.rstrip("\n")
        if not raw.strip():
            result.append("")
            continue

        speaker_idx = (speaker_idx % num_speakers) + 1
        result.append(f"[SPK{speaker_idx}] {raw}")

    return result


def ensure_text_model_loaded(app) -> Optional["SentenceTransformer"]:
    """Ładuje (a w razie potrzeby pobiera) model embeddingsów."""
    if not HAS_NLP_LIBS:
        messagebox.showerror(
            "Brak bibliotek",
            "Do metod diarizacji z embeddingsami potrzebne są biblioteki:\n\n"
            "pip install sentence-transformers scikit-learn"
        )
        return None

    if getattr(app, "emb_model", None) is not None:
        return app.emb_model

    try:
        if hasattr(app, 'log'):
            app.log(f"Ładuję model embeddings: {MODEL_NAME} (pierwsze użycie)")
        app.emb_model = SentenceTransformer(MODEL_NAME)
        if hasattr(app, 'log'):
            app.log('Model embeddings załadowany.')
        return app.emb_model
    except Exception as e:
        messagebox.showerror(
            "Błąd ładowania modelu",
            f"Nie udało się załadować modelu {MODEL_NAME}:\n{e}"
        )
        return None


def diarize_with_embeddings_fixed(lines: List[str], num_speakers: int, model) -> List[str]:
    """Diarizacja z embeddingsami – liczba mówców podana ręcznie."""
    non_empty_lines = []
    line_indices = []
    for idx, line in enumerate(lines):
        if line.strip():
            non_empty_lines.append(line)
            line_indices.append(idx)

    if not non_empty_lines:
        return lines

    if len(non_empty_lines) < num_speakers:
        num_speakers = max(1, len(non_empty_lines))

    embeddings = model.encode(non_empty_lines, convert_to_numpy=True, show_progress_bar=False)
    km = KMeans(n_clusters=num_speakers, random_state=0, n_init=10)
    labels = km.fit_predict(embeddings)

    result = list(lines)
    cluster_to_spk = {}
    next_spk_id = 1

    for idx, cluster in zip(line_indices, labels):
        line = lines[idx].rstrip("\n")
        if cluster not in cluster_to_spk:
            cluster_to_spk[cluster] = next_spk_id
            next_spk_id += 1
        spk_id = cluster_to_spk[cluster]
        result[idx] = f"[SPK{spk_id}] {line}"

    return result


def diarize_with_embeddings_auto(lines: List[str], max_speakers: int, model) -> List[str]:
    """Diarizacja z embeddingsami – liczba mówców zgadywana (2..max_speakers)."""
    non_empty_lines = []
    line_indices = []
    for idx, line in enumerate(lines):
        if line.strip():
            non_empty_lines.append(line)
            line_indices.append(idx)

    if not non_empty_lines:
        return lines

    if len(non_empty_lines) < 2:
        return diarize_alternating(lines, num_speakers=1)

    embeddings = model.encode(non_empty_lines, convert_to_numpy=True, show_progress_bar=False)

    best_k = None
    best_score = -1.0
    max_k = min(max_speakers, len(non_empty_lines))

    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    if best_k is None:
        best_k = 2
        if len(non_empty_lines) < 2:
            best_k = 1

    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = km.fit_predict(embeddings)

    result = list(lines)
    cluster_to_spk = {}
    next_spk_id = 1

    for idx, cluster in zip(line_indices, labels):
        line = lines[idx].rstrip("\n")
        if cluster not in cluster_to_spk:
            cluster_to_spk[cluster] = next_spk_id
            next_spk_id += 1
        spk_id = cluster_to_spk[cluster]
        result[idx] = f"[SPK{spk_id}] {line}"

    return result


# ========= OBSŁUGA WHISPER =========

def ensure_whisper_model_loaded(app) -> Optional["whisper.Whisper"]:
    """Ładuje model Whispera (ten sam dla transkrypcji i diarizacji)."""
    if not HAS_WHISPER:
        messagebox.showerror(
            "Brak Whispera",
            "Do transkrypcji audio potrzebny jest pakiet openai-whisper:\n\n"
            "pip install openai-whisper\n\n"
            "oraz ffmpeg:\n\n"
            "sudo apt install ffmpeg"
        )
        return None

    model_name = app.whisper_model_var.get()
    if getattr(app, "whisper_model", None) is not None and app.current_whisper_model_name == model_name:
        return app.whisper_model

    try:
        if hasattr(app, 'log'):
            app.log(f"Ładuję model Whisper: {model_name} (może potrwać)")
        app.whisper_model = whisper.load_model(model_name)
        app.current_whisper_model_name = model_name
        app.update_status_label()
        if hasattr(app, 'log'):
            app.log('Model Whisper załadowany.')
        return app.whisper_model
    except Exception as e:
        messagebox.showerror(
            "Błąd ładowania Whisper",
            f"Nie udało się załadować modelu '{model_name}':\n{e}"
        )
        return None


def format_time(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ========= PYANNOTE – DIARYZACJA PO GŁOSIE =========

def ensure_pyannote_pipeline_loaded(app) -> Optional["Pipeline"]:
    """
    Ładuje pipeline pyannote do diarizacji głosowej.
    Wymaga tokena HF (GUI/env). Zapisuje token do pliku.
    """
    if not HAS_PYANNOTE:
        messagebox.showerror(
            "Brak pyannote.audio",
            "Do diarizacji po barwie głosu potrzebny jest pakiet pyannote.audio:\n\n"
            "pip install pyannote.audio torch"
        )
        return None

    if getattr(app, "pyannote_pipeline", None) is not None:
        return app.pyannote_pipeline

    token = (
        app.pyannote_token_var.get().strip()
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        messagebox.showerror(
            "Brak tokena HuggingFace",
            "Przejdź do zakładki 'Ustawienia' i podaj token HF\n"
            "lub ustaw zmienną środowiskową HF_TOKEN."
        )
        return None

    try:
        if hasattr(app, 'log'):
            app.log('Ładuję pipeline pyannote/speaker-diarization-3.1 (może potrwać)...')
        app.save_hf_token(token)
        os.environ["HF_TOKEN"] = token
        app.pyannote_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        app.update_status_label()
        if hasattr(app, 'log'):
            app.log('Pipeline pyannote załadowany.')
        return app.pyannote_pipeline
    except Exception as e:
        messagebox.showerror(
            "Błąd ładowania pyannote",
            f"Nie udało się załadować pipeline pyannote:\n{e}"
        )
        return None


def diarize_audio_with_pyannote(pipeline: "Pipeline", audio_path: str):
    """
    Zwraca listę segmentów po diarizacji po głosie.

    Aby uniknąć błędów typu:
    "file resulted in XXXX samples instead of the expected YYYY samples"
    przekodowujemy wejściowe audio do WAV 16 kHz mono za pomocą ffmpeg
    i diarizację robimy na tym tymczasowym pliku.

    Działa zarówno z pyannote.audio 3.x (pipeline zwraca Annotation),
    jak i 4.x (pipeline zwraca DiarizeOutput, a Annotation siedzi
    w output.speaker_diarization).
    """
    tmpdir = tempfile.mkdtemp(prefix="pyannote_")
    wav_path = os.path.join(tmpdir, "audio_16k.wav")

    try:
        # ffmpeg -y -i input -ac 1 -ar 16000 output.wav
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ac", "1",
            "-ar", "16000",
            wav_path,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            msg = proc.stderr.decode(errors="ignore")[:4000]
            raise RuntimeError(f"ffmpeg error while converting audio:\n{msg}")

        output = pipeline(wav_path)
        ann = getattr(output, "speaker_diarization", output)

        segments = []
        for turn, _, speaker in ann.itertracks(yield_label=True):
            segments.append(
                (
                    float(turn.start),
                    float(turn.end),
                    str(speaker),
                )
            )
        return segments

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def assign_speakers_by_time(whisper_segments, speaker_segments):
    """
    Łączy segmenty tekstowe (Whisper) z segmentami mówców (pyannote)
    na podstawie nakładania się w czasie.
    """
    result = []

    for ws, we, text in whisper_segments:
        best_speaker = None
        best_overlap = 0.0

        for ss, se, spk in speaker_segments:
            overlap_start = max(ws, ss)
            overlap_end = min(we, se)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk

        if best_speaker is None:
            best_speaker = "UNKNOWN"

        result.append((best_speaker, ws, we, text))

    return result


# ========= GUI =========

class DiarizationApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1200x780")

        self.emb_model = None
        self.whisper_model = None
        self.current_whisper_model_name = None
        self.pyannote_pipeline = None
        self.audio_path = None

        self.method_display_to_code = {
            "Szybka – naprzemienna": "alternating",
            "Dokładniejsza – embeddings (liczba mówców)": "embeddings_fixed",
            "Najlepsza – embeddings (auto liczba mówców)": "embeddings_auto",
        }

        self.pyannote_token_var = tk.StringVar(value="")
        self.whisper_model_var = tk.StringVar(value="small")
        self.whisper_lang_var = tk.StringVar(value="pl")
        self.status_var = tk.StringVar(value="")

        self._pending_logs = []
        self.log('Aplikacja uruchomiona.')

        self.load_saved_hf_token()

        self._create_widgets()
        self._create_menu()
        self.update_status_label()

    # ----- HF token persistence -----
    def load_saved_hf_token(self):
        try:
            if os.path.exists(HF_TOKEN_FILE):
                with open(HF_TOKEN_FILE, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                if token:
                    self.pyannote_token_var.set(token)
        except Exception:
            pass

    def save_hf_token(self, token: str):
        try:
            with open(HF_TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(token.strip())
        except Exception:
            pass

    # ====== WIDŻETY / UKŁAD ======

    def _create_widgets(self):
        # Notebook z zakładkami
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        main_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        info_tab = ttk.Frame(notebook)

        notebook.add(main_tab, text="Główne")
        notebook.add(settings_tab, text="Ustawienia")
        notebook.add(info_tab, text="Info")

        self._create_main_tab(main_tab)
        self._create_settings_tab(settings_tab)
        self._create_info_tab(info_tab)

        # Pasek statusu na dole
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_info_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="Informacje o programie")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info = build_info_text()

        txt = scrolledtext.ScrolledText(frame, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True)

        txt.insert("1.0", info)
        txt.configure(state="disabled")

        
    def _create_main_tab(self, parent):
        # ====== SEKCJA: AUDIO ======
        audio_frame = ttk.LabelFrame(parent, text="Audio")
        audio_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(audio_frame, text="Język (auto / pl / en / ...):").pack(side=tk.LEFT, padx=5)

        # Combobox z popularnymi kodami językowymi; wpis własny też jest OK
        self.whisper_lang_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.whisper_lang_var,
            state="normal",
            values=["auto", "pl", "en", "de", "fr", "es", "ru", "uk", "ja"],
            width=6,
        )
        self.whisper_lang_combo.pack(side=tk.LEFT, padx=5)

        load_audio_btn = ttk.Button(
            audio_frame,
            text="Wczytaj plik audio",
            command=self.on_load_audio_clicked,
        )
        load_audio_btn.pack(side=tk.LEFT, padx=10)

        self.audio_file_label = ttk.Label(audio_frame, text="Brak wybranego pliku audio.")
        self.audio_file_label.pack(side=tk.LEFT, padx=10)

        # ====== SEKCJA: TRANSKRYPCJA / DIARYZACJA (WHISPER) ======
        trans_frame = ttk.LabelFrame(parent, text="Transkrypcja / diaryzacja zaawansowana")
        trans_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(trans_frame, text="Model Whisper:").pack(side=tk.LEFT, padx=5)

        whisper_model_combo = ttk.Combobox(
            trans_frame,
            textvariable=self.whisper_model_var,
            state="readonly",
            values=["tiny", "base", "small", "medium", "large"],
            width=10,
        )
        whisper_model_combo.pack(side=tk.LEFT, padx=5)
        whisper_model_combo.bind("<<ComboboxSelected>>", self.on_whisper_model_change)

        ttk.Label(
            trans_frame,
            text="(tiny/base/small – szybsze, medium – dokładniejsze, large – najlepsze)"
        ).pack(side=tk.LEFT, padx=5)

        transcribe_btn = ttk.Button(
            trans_frame,
            text="Transkrybuj audio (Whisper)",
            command=self.on_transcribe_audio_clicked,
        )
        transcribe_btn.pack(side=tk.LEFT, padx=10)

        diarize_voice_btn = ttk.Button(
            trans_frame,
            text="Transkrybuj + diaryzacja",
            command=self.on_transcribe_and_diarize_voice_clicked,
        )
        diarize_voice_btn.pack(side=tk.LEFT, padx=10)

        # ====== SEKCJA: DIARYZACJA TEKSTU ======
        settings_frame = ttk.LabelFrame(parent, text="Diaryzacja podstawowa (na bazie transkryptu)")
        settings_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(settings_frame, text="Metoda:").pack(side=tk.LEFT, padx=5)

        self.method_var = tk.StringVar(value="Szybka – naprzemienna")
        method_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.method_var,
            state="readonly",
            values=list(self.method_display_to_code.keys()),
            width=40,
        )
        method_combo.pack(side=tk.LEFT, padx=5)
        method_combo.bind("<<ComboboxSelected>>", self.on_method_change)

        self.speakers_label = ttk.Label(settings_frame, text="Liczba mówców:")
        self.speakers_var = tk.IntVar(value=2)
        self.speakers_spin = ttk.Spinbox(
            settings_frame,
            from_=1,
            to=20,
            textvariable=self.speakers_var,
            width=5,
        )

        self.max_speakers_label = ttk.Label(settings_frame, text="Max mówców (auto):")
        self.max_speakers_var = tk.IntVar(value=5)
        self.max_speakers_spin = ttk.Spinbox(
            settings_frame,
            from_=2,
            to=20,
            textvariable=self.max_speakers_var,
            width=5,
        )

        diarize_button = ttk.Button(
            settings_frame,
            text="Diaryzuj tekst (bez audio)",
            command=self.on_diarize_clicked,
        )
        diarize_button.pack(side=tk.RIGHT, padx=10)

        self.model_info_label = ttk.Label(
            settings_frame,
            text="Metody z embeddingsami pobiorą model tekstowy przy pierwszym użyciu.",
        )
        self.model_info_label.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

        # ====== DWA OKNA TEKSTOWE (transkrypt / wynik) ======
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        ttk.Label(
            left_frame,
            text="Wejściowy tekst (transkrypt – z pliku lub audio):",
        ).pack(anchor="w")

        self.input_text = tk.Text(left_frame, wrap="word", undo=True)
        self.input_text.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        ttk.Label(
            right_frame,
            text="Wynik diaryzacji tekstu:",
        ).pack(anchor="w")

        self.output_text = tk.Text(right_frame, wrap="word", state="disabled")
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # ====== LOGI / DZIAŁANIE PROGRAMU ======
        log_frame = ttk.LabelFrame(parent, text="Logi działania")
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=10, pady=5)

        top_log_bar = ttk.Frame(log_frame)
        top_log_bar.pack(side=tk.TOP, fill=tk.X)

        clear_btn = ttk.Button(top_log_bar, text="Wyczyść logi", command=self.clear_logs)
        clear_btn.pack(side=tk.RIGHT, padx=5, pady=2)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ustaw jako read-only
        self.log_text.configure(state="disabled")

        # zrzut zaległych logów z czasu inicjalizacji
        self._flush_pending_logs()

        self.on_method_change()

    def _create_settings_tab(self, parent):
        # --- HF token dla pyannote (jedyna sekcja ustawień) ---
        pyannote_frame = ttk.LabelFrame(parent, text="Token HuggingFace (pyannote)")
        pyannote_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(pyannote_frame, text="HF token:").pack(side=tk.LEFT, padx=5)
        pyannote_token_entry = ttk.Entry(
            pyannote_frame,
            textvariable=self.pyannote_token_var,
            width=50,
            show="*",
        )
        pyannote_token_entry.pack(side=tk.LEFT, padx=5)

        save_token_btn = ttk.Button(
            pyannote_frame,
            text="Zapisz token HF",
            command=self.on_save_token_clicked,
        )
        save_token_btn.pack(side=tk.LEFT, padx=10)

        ttk.Label(
            pyannote_frame,
            text="Token będzie zapisany w ~/.pyannote_hf_token i używany przy diaryzacji.",
        ).pack(side=tk.LEFT, padx=5)

    # ======= MENU =======

    def _create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)

        filemenu.add_command(label="Otwórz tekst z pliku...", command=self.open_file_text)
        filemenu.add_command(label="Zapisz tekst wejściowy...", command=self.save_input)
        filemenu.add_command(label="Zapisz wynik diaryzacji...", command=self.save_output)
        filemenu.add_separator()
        filemenu.add_command(label="Zakończ", command=self.quit)

        menubar.add_cascade(label="Plik", menu=filemenu)
        self.config(menu=menubar)

    # ======= HELPERY GUI =======

    def get_current_text_diarization_code(self) -> str:
        return self.method_display_to_code.get(self.method_var.get(), "alternating")

    def get_whisper_quality_desc(self, model_name: str) -> str:
        mapping = {
            "tiny": "najszybszy / najsłabszy",
            "base": "bardzo szybki",
            "small": "szybki (domyślny)",
            "medium": "dokładniejszy",
            "large": "najlepsza jakość",
        }
        return mapping.get(model_name, "niestandardowy")

    def update_status_label(self):
        whisper_model = self.whisper_model_var.get()
        whisper_desc = self.get_whisper_quality_desc(whisper_model)

        text_method_display = getattr(self, "method_var", tk.StringVar(value="")).get()
        text_method_code = self.get_current_text_diarization_code() if hasattr(self, "method_var") else ""

        if HAS_PYANNOTE:
            voice_info = "pyannote/speaker-diarization-3.1"
        else:
            voice_info = "brak (pyannote.audio nie zainstalowane)"

        self.status_var.set(
            f"Transkrypcja: Whisper-{whisper_model} ({whisper_desc}) | "
            f"Diarizacja tekstu: {text_method_display} [{text_method_code}] | "
            f"Diarizacja po głosie: {voice_info}"
        )

    # ======= LOGI =======

    def log(self, message: str):
        """Dopisuje wpis do okna logów (z timestampem)."""
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception:
            ts = ""
        line = f"[{ts}] {message}".strip()

        # jeśli GUI logów nie gotowe – buforuj
        if not hasattr(self, "log_text") or self.log_text is None:
            if not hasattr(self, "_pending_logs"):
                self._pending_logs = []
            self._pending_logs.append(line)
            return

        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _flush_pending_logs(self):
        """Wypisuje zaległe logi z bufora do widgetu (po jego utworzeniu)."""
        pending = getattr(self, "_pending_logs", [])
        if not pending or not hasattr(self, "log_text") or self.log_text is None:
            return
        self.log_text.configure(state="normal")
        for line in pending:
            self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self._pending_logs = []

    def clear_logs(self):
        if not hasattr(self, "log_text") or self.log_text is None:
            self._pending_logs = []
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    # ======= ZMIANA USTAWIEŃ =======

    def on_method_change(self, event=None):
        """Reaguje na zmianę metody diaryacji tekstu (pokazuje właściwe parametry)."""
        code = self.get_current_text_diarization_code()

        # Ukryj wszystkie kontrolki parametrów
        if hasattr(self, "speakers_label"):
            self.speakers_label.pack_forget()
        if hasattr(self, "speakers_spin"):
            self.speakers_spin.pack_forget()
        if hasattr(self, "max_speakers_label"):
            self.max_speakers_label.pack_forget()
        if hasattr(self, "max_speakers_spin"):
            self.max_speakers_spin.pack_forget()

        # Pokaż właściwe kontrolki
        parent = getattr(self, "speakers_label", None).master if hasattr(self, "speakers_label") else None
        if parent is not None:
            if code in ("alternating", "embeddings_fixed"):
                self.speakers_label.pack(in_=parent, side=tk.LEFT, padx=5)
                self.speakers_spin.pack(in_=parent, side=tk.LEFT, padx=2)
            elif code == "embeddings_auto":
                self.max_speakers_label.pack(in_=parent, side=tk.LEFT, padx=5)
                self.max_speakers_spin.pack(in_=parent, side=tk.LEFT, padx=2)

        self.update_status_label()

    def on_whisper_model_change(self, event=None):
        """Reaguje na zmianę modelu Whisper – wymusza przeładowanie przy następnym użyciu."""
        self.current_whisper_model_name = None
        self.update_status_label()

    def on_save_token_clicked(self):
        """Zapis tokena HF do pliku i ustawienie w env (dla pyannote)."""
        token = self.pyannote_token_var.get().strip()
        if not token:
            messagebox.showwarning("Brak tokena", "Podaj token HF, aby go zapisać.")
            return
        self.save_hf_token(token)
        os.environ["HF_TOKEN"] = token
        messagebox.showinfo("Zapisano token", "Token HF został zapisany.")
        self.update_status_label()

    # ======= DIARYZACJA TEKSTU =======

    def on_diarize_clicked(self):
        text = self.input_text.get("1.0", tk.END)
        if not text.strip():
            self.log("Diarizacja tekstu: brak tekstu – przerwano.")
            messagebox.showwarning("Brak tekstu", "Wklej transkrypt lub zrób transkrypcję audio.")
            return

        self.log("Diarizacja tekstu: start.")
        lines = text.splitlines()
        method_code = self.get_current_text_diarization_code()
        self.log(f"Diarizacja tekstu: metoda '{self.method_var.get()}'")

        try:
            if method_code == "alternating":
                num_speakers = max(1, self.speakers_var.get())
                result_lines = diarize_alternating(lines, num_speakers)

            elif method_code == "embeddings_fixed":
                model = ensure_text_model_loaded(self)
                if model is None:
                    return
                num_speakers = max(1, self.speakers_var.get())
                result_lines = diarize_with_embeddings_fixed(lines, num_speakers, model)

            elif method_code == "embeddings_auto":
                model = ensure_text_model_loaded(self)
                if model is None:
                    return
                max_speakers = max(2, self.max_speakers_var.get())
                result_lines = diarize_with_embeddings_auto(lines, max_speakers, model)

            else:
                messagebox.showerror("Błąd", f"Nieznana metoda diarizacji: {method_code}")
                self.log(f"Diarizacja tekstu: nieznana metoda: {method_code}")
                return

        except Exception as e:
            messagebox.showerror("Błąd diarizacji", f"Wystąpił błąd:\n{e}")
            self.log(f"Diarizacja tekstu: błąd: {e}")
            return

        self._set_output_text("\n".join(result_lines))
        self.log("Diarizacja tekstu zakończona – wynik w prawym panelu.")

    def _set_output_text(self, content: str):
        """Ustawia treść w prawym panelu wyniku (read-only)."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", content)
        self.output_text.config(state="disabled")

    # ======= OBSŁUGA TEKSTU Z PLIKU / ZAPIS =======

    def open_file_text(self):
        filename = filedialog.askopenfilename(
            title="Wybierz plik z transkryptem",
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")],
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(filename, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                messagebox.showerror("Błąd odczytu", f"Nie udało się odczytać pliku:\n{e}")
                return
        except Exception as e:
            messagebox.showerror("Błąd odczytu", f"Nie udało się odczytać pliku:\n{e}")
            return

        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", content)
        self.log(f"Wczytano transkrypt z pliku: {filename}")

    def save_input(self):
        input_text = self.input_text.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showwarning("Brak danych", "Brak tekstu wejściowego do zapisania.")
            return

        filename = filedialog.asksaveasfilename(
            title="Zapisz tekst wejściowy",
            defaultextension=".txt",
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")],
        )
        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(input_text)
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")

    def save_output(self):
        diarized_text = self.output_text.get("1.0", tk.END)
        if not diarized_text.strip():
            messagebox.showwarning("Brak danych", "Brak wyniku diarizacji do zapisania.")
            return

        filename = filedialog.asksaveasfilename(
            title="Zapisz wynik diarizacji",
            defaultextension=".txt",
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")],
        )
        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(diarized_text)
        except Exception as e:
            messagebox.showerror("Błąd zapisu", f"Nie udało się zapisać pliku:\n{e}")

    # ======= AUDIO: WCZYTYWANIE I TRANSKRYPCJA =======

    def on_load_audio_clicked(self):
        filename = filedialog.askopenfilename(
            title="Wybierz plik audio",
            filetypes=[
                ("Pliki audio", "*.m4a *.mp3 *.wav *.flac *.ogg"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not filename:
            return

        self.audio_path = filename
        self.audio_file_label.config(text=f"Wybrano: {filename}")
        self.log(f"Wczytano plik audio: {filename}")


    def on_transcribe_audio_clicked(self):
        if not self.audio_path:
            self.log("Transkrypcja: brak pliku audio – przerwano.")
            messagebox.showwarning("Brak audio", "Najpierw wybierz plik audio.")
            return

        model = ensure_whisper_model_loaded(self)
        if model is None:
            return

        self.log(f"Transkrypcja: używam modelu Whisper '{self.whisper_model_var.get()}'")

        lang_raw = self.whisper_lang_var.get().strip()
        lang = None if (not lang_raw or lang_raw.lower() == "auto") else lang_raw

        try:
            self.log("Transkrypcja: przetwarzam audio...")
            self.update_idletasks()
            result = model.transcribe(self.audio_path, language=lang, verbose=False)
        except Exception as e:
            messagebox.showerror(
                "Błąd transkrypcji",
                f"Nie udało się przetworzyć pliku audio:\n{e}"
            )
            self.log(f"Transkrypcja: błąd: {e}")
            return

        segments = result.get("segments", [])
        if not segments:
            messagebox.showwarning("Brak segmentów", "Whisper nie zwrócił segmentów.")
            self.log("Transkrypcja: brak segmentów w wyniku.")
            return

        lines = []
        for seg in segments:
            start = format_time(seg.get("start", 0.0))
            end = format_time(seg.get("end", 0.0))
            text = seg.get("text", "").strip()
            if not text:
                continue
            lines.append(f"[{start}–{end}] {text}")

        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", "\n".join(lines))
        self.log("Transkrypcja zakończona – wynik wklejony do pola wejściowego.")

        messagebox.showinfo(
            "Transkrypcja zakończona",
            "Transkrypcja audio została wczytana do pola tekstowego.\n"
            "Możesz teraz użyć 'Diarizuj tekst' lub diarizacji po głosie."
        )


    def on_transcribe_and_diarize_voice_clicked(self):
        if not self.audio_path:
            self.log("Diarizacja po głosie: brak pliku audio – przerwano.")
            messagebox.showwarning("Brak audio", "Najpierw wybierz plik audio.")
            return

        model = ensure_whisper_model_loaded(self)
        if model is None:
            return

        self.log(f"Diarizacja po głosie: transkrybuję Whisper '{self.whisper_model_var.get()}'")

        lang_raw = self.whisper_lang_var.get().strip()
        lang = None if (not lang_raw or lang_raw.lower() == "auto") else lang_raw

        try:
            self.log("Diarizacja po głosie: wykonuję transkrypcję Whisper...")
            self.update_idletasks()
            result = model.transcribe(self.audio_path, language=lang, verbose=False)
        except Exception as e:
            messagebox.showerror(
                "Błąd transkrypcji",
                f"Nie udało się przetworzyć pliku audio (Whisper):\n{e}"
            )
            self.log(f"Diarizacja po głosie: błąd transkrypcji: {e}")
            return

        segs = result.get("segments", [])
        if not segs:
            messagebox.showwarning("Brak segmentów", "Whisper nie zwrócił segmentów.")
            self.log("Diarizacja po głosie: brak segmentów Whisper.")
            return

        whisper_segments = []
        plain_text_lines = []

        for seg in segs:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            txt = seg.get("text", "").strip()
            if not txt:
                continue
            whisper_segments.append((s, e, txt))
            plain_text_lines.append(f"[{format_time(s)}–{format_time(e)}] {txt}")

        pipeline = ensure_pyannote_pipeline_loaded(self)
        if pipeline is None:
            return

        try:
            self.log("Diarizacja po głosie: uruchamiam pyannote (ffmpeg→WAV 16k mono + pipeline)...")
            self.update_idletasks()
            speaker_segments = diarize_audio_with_pyannote(pipeline, self.audio_path)
        except Exception as e:
            messagebox.showerror(
                "Błąd diarizacji po głosie",
                f"Nie udało się przeprowadzić diarizacji pyannote:\n{e}"
            )
            self.log(f"Diarizacja po głosie: błąd pyannote: {e}")
            return

        if not speaker_segments:
            messagebox.showwarning("Brak segmentów mówców", "pyannote nie zwrócił segmentów mówców.")
            self.log("Diarizacja po głosie: brak segmentów mówców z pyannote.")
            return

        combined = assign_speakers_by_time(whisper_segments, speaker_segments)

        speaker_map = {}
        next_id = 1
        diarized_lines = []

        for spk_label, s, e, txt in combined:
            if spk_label not in speaker_map:
                speaker_map[spk_label] = next_id
                next_id += 1
            spk_id = speaker_map[spk_label]
            diarized_lines.append(
                f"[SPK{spk_id}][{format_time(s)}–{format_time(e)}] {txt}"
            )

        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", "\n".join(plain_text_lines))

        self._set_output_text("\n".join(diarized_lines))
        self.log("Diarizacja po głosie zakończona – wynik w prawym panelu.")

        messagebox.showinfo(
            "Transkrypcja + diarizacja po głosie",
            "Zakończono transkrypcję i diarizację po barwie głosu.\n"
            "Lewy panel: czysty transkrypt.\n"
            "Prawy panel: diarizacja po głosie (SPK1, SPK2, ...)."
        )


if __name__ == "__main__":
    app = DiarizationApp()
    app.mainloop()
