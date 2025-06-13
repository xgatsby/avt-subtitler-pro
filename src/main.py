# =======================================================================
# AVT SUBTITLER PRO - FINAL ALL-IN-ONE SCRIPT (No spaCy)
# Version: 6.0.1 (Final Logic and Structure)
# =======================================================================

# --- Bagian 1: Impor & Konfigurasi Awal ---
import os, sys, json, re, time, logging, datetime, shutil
from typing import List, Dict, Tuple, Optional, frozenset
from dataclasses import dataclass
import torch
import moviepy.editor as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForSpeechSeq2Seq
from kaggle_secrets import UserSecretsClient
import pysrt
import yaml
import argparse

# --- Bagian 2: Kelas-Kelas Generator Konten (Tahap 1) ---

class ContentConfig:
    """
    Konfigurasi untuk path data, model, dan output yang digunakan dalam pembuatan konten.
    """
    DATASET_PATH: str = "/kaggle/input/avt-subtitler-pro-assets"
    VIDEO_INPUT_PATH: str = os.path.join(DATASET_PATH, "input.mp4")
    MAPPING_JSON_PATH: str = os.path.join(DATASET_PATH, "mapping.json") # Peta istilah khusus EN -> ID
    RAW_SRT_OUTPUT_PATH: str = "/kaggle/working/raw_subtitle.srt" # Output SRT mentah setelah transkripsi dan terjemahan awal
    LOG_FILE_PATH: str = "/kaggle/working/avt_subtitler_pro.log"
    WHISPER_MODEL: str = "xgatsby/whisper-large-v3-avt-workshop" # Model Whisper untuk transkripsi ASR
    TRANSLATION_MODEL: str = "xgatsby/opus-mt-en-id-avt" # Model NMT untuk terjemahan EN-ID
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu" # Perangkat untuk inferensi model (GPU jika ada)

class ContentGenerator:
    """
    Mengelola proses pembuatan konten mentah: transkripsi audio dan terjemahan awal.
    """
    def __init__(self) -> None:
        """
        Inisialisasi ContentGenerator dengan konfigurasi dan komponen terkait.
        """
        self.cfg: ContentConfig = ContentConfig()
        self.translation_engine: ContentGenerator.TranslationEngine = self.TranslationEngine(self.cfg)
        self.transcriber: ContentGenerator.AudioTranscriber = self.AudioTranscriber(self.cfg)

    def run(self) -> None:
        """
        Menjalankan alur kerja pembuatan konten mentah:
        1. Mendapatkan token Hugging Face.
        2. Memuat model transkripsi.
        3. Mentranskripsi audio dari video input.
        4. Memuat model terjemahan dan peta istilah.
        5. Memproses segmen transkripsi menjadi kalimat logis, menerjemahkannya, dan menulis file SRT mentah.
        """
        logging.info("=== RAW CONTENT GENERATOR STARTING ===")
        hf_token: Optional[str] = self._get_hf_token()

        self.transcriber.load_model(hf_token)
        transcribed_segments: List[Dict] = self.transcriber.transcribe(self.cfg.VIDEO_INPUT_PATH)
        if not transcribed_segments:
            logging.error("Transcription produced no segments. Aborting.")
            raise RuntimeError("Transcription produced no segments.")

        self.translation_engine.load(hf_token)
        self.translation_engine.process_and_write_srt(transcribed_segments, self.cfg.RAW_SRT_OUTPUT_PATH)
        logging.info(f"Raw subtitle file created at {self.cfg.RAW_SRT_OUTPUT_PATH}")
    
    def _get_hf_token(self) -> Optional[str]:
        """
        Mengambil token Hugging Face dari Kaggle Secrets.
        Returns:
            Optional[str]: Token Hugging Face jika ditemukan, None jika tidak.
        """
        try:
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:
            logging.warning(f"Could not get HF_TOKEN from Kaggle Secrets: {e}. Proceeding without token.")
            return None

    class AudioTranscriber:
        """
        Mengelola transkripsi audio menggunakan model Whisper.
        """
        def __init__(self, config: ContentConfig) -> None:
            """
            Inisialisasi AudioTranscriber.
            Args:
                config (ContentConfig): Konfigurasi yang digunakan.
            """
            self.cfg: ContentConfig = config
            self.pipe: Optional[pipeline] = None # Pipeline Hugging Face untuk ASR

        def load_model(self, token: Optional[str]) -> None:
            """
            Memuat model Whisper dan prosesor terkait.
            Args:
                token (Optional[str]): Token Hugging Face untuk mengakses model privat jika diperlukan.
            """
            logging.info(f"Loading Whisper model '{self.cfg.WHISPER_MODEL}'...")
            model: AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.cfg.WHISPER_MODEL,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=token
            )
            model.to(self.cfg.DEVICE)
            processor: AutoProcessor = AutoProcessor.from_pretrained(self.cfg.WHISPER_MODEL, token=token)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=torch.float16,
                device=self.cfg.DEVICE
            )
            logging.info("Whisper model loaded successfully.")

        def transcribe(self, video_path: str) -> List[Dict]:
            """
            Mentranskripsi audio dari file video yang diberikan.
            Args:
                video_path (str): Path ke file video.
            Returns:
                List[Dict]: Daftar segmen transkripsi, masing-masing berisi 'text' dan 'timestamp'.
            """
            if not self.pipe:
                logging.error("Transcription pipe not initialized. Call load_model first.")
                raise RuntimeError("Transcription pipe not initialized.")
            logging.info(f"Starting transcription for {video_path}...")
            result: Dict = self.pipe(video_path, generate_kwargs={"language": "english"})
            logging.info("Transcription finished.")
            return result.get("chunks", [])

    class TranslationEngine:
        """
        Mengelola proses terjemahan teks dan penulisan file SRT.
        """
        def __init__(self, config: ContentConfig) -> None:
            """
            Inisialisasi TranslationEngine.
            Args:
                config (ContentConfig): Konfigurasi yang digunakan.
            """
            self.cfg: ContentConfig = config
            self.mapping: Dict[str, str] = {} # Peta istilah khusus (dari output model ID -> pengganti ID yang diinginkan)
            self.model: Optional[AutoModelForSeq2SeqLM] = None
            self.tokenizer: Optional[AutoTokenizer] = None
        
        def load(self, token: Optional[str]) -> None:
            """
            Memuat model terjemahan, tokenizer, dan peta istilah khusus.
            Peta istilah diurutkan berdasarkan panjang kunci (descending) untuk prioritas pencocokan yang lebih panjang.
            Args:
                token (Optional[str]): Token Hugging Face.
            """
            logging.info(f"Loading translation model '{self.cfg.TRANSLATION_MODEL}' and mapping file...")
            with open(self.cfg.MAPPING_JSON_PATH, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            # Urutkan mapping berdasarkan panjang kunci (descending) untuk memastikan istilah yang lebih panjang (lebih spesifik)
            # dicocokkan terlebih dahulu.
            self.mapping = dict(sorted(self.mapping.items(), key=lambda item: len(item[0]), reverse=True))

            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TRANSLATION_MODEL, token=token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.TRANSLATION_MODEL, token=token).to(self.cfg.DEVICE)
            logging.info("Translation Engine models and mapping loaded successfully.")

        def _translate_sentence(self, sentence_text: str) -> str:
            """
            Menerjemahkan satu kalimat dari Bahasa Inggris ke Bahasa Indonesia dan menerapkan pemetaan pasca-terjemahan.
            Args:
                sentence_text (str): Kalimat dalam Bahasa Inggris untuk diterjemahkan.
            Returns:
                str: Kalimat yang telah diterjemahkan ke Bahasa Indonesia dan diproses dengan peta istilah.
            """
            if not sentence_text.strip():
                return ""
            if not self.model or not self.tokenizer:
                logging.error("Translation model/tokenizer not loaded.")
                raise RuntimeError("Translation model/tokenizer not loaded.")

            # Langkah 1: Terjemahan utama dari EN ke ID
            inputs = self.tokenizer(sentence_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.cfg.DEVICE)
            translated_tokens = self.model.generate(**inputs, max_length=512)
            indonesian_translation: str = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            # Langkah 2: Aplikasi pemetaan pasca-terjemahan
            # Iterasi melalui peta istilah yang telah diurutkan (self.mapping).
            # Kunci dalam self.mapping adalah istilah Bahasa Indonesia yang mungkin dihasilkan oleh model NMT,
            # dan nilainya adalah pengganti Bahasa Indonesia yang diinginkan (lebih akurat atau sesuai konteks).
            # Ini berguna untuk memperbaiki terjemahan istilah teknis, nama, atau frasa umum.
            processed_translation: str = indonesian_translation
            for term_to_find_in_model_output, desired_replacement in self.mapping.items():
                # Penggantian case-insensitive menggunakan regex.
                # re.escape memastikan 'term_to_find_in_model_output' diperlakukan sebagai string literal.
                processed_translation = re.sub(
                    re.escape(term_to_find_in_model_output),
                    desired_replacement,
                    processed_translation,
                    flags=re.IGNORECASE
                )
            return processed_translation

        def process_and_write_srt(self, segments: List[Dict], output_path: str) -> None:
            """
            Memproses segmen transkripsi menjadi kalimat logis, menerjemahkannya, dan menulis hasilnya ke file SRT.
            Args:
                segments (List[Dict]): Daftar segmen dari Whisper, masing-masing dengan 'text' dan 'timestamp'.
                output_path (str): Path untuk menyimpan file SRT yang dihasilkan.
            """
            if not segments:
                logging.warning("No segments to process for SRT generation. Writing an empty SRT file.")
                with open(output_path, 'w', encoding='utf-8') as f: # Ensure UTF-8 for SRT
                    f.write("")
                return

            # --- Logika Segmentasi Kalimat ---
            # Tujuan: Menggabungkan segmen-segmen pendek dari Whisper menjadi kalimat-kalimat yang lebih logis
            # sebelum terjemahan. Ini meningkatkan kualitas terjemahan karena model NMT bekerja lebih baik
            # dengan konteks kalimat penuh.
            #
            # Strategi:
            # 1. Akumulasi teks dan timestamp dari segmen Whisper.
            # 2. Deteksi akhir kalimat berdasarkan:
            #    a. Tanda baca (".", "?", "!").
            #    b. Merupakan segmen terakhir dari input.
            # 3. Setelah kalimat logis terbentuk:
            #    a. Normalisasi spasi (terutama setelah tanda baca).
            #    b. Simpan teks kalimat, timestamp awal dan akhir gabungan, serta jumlah segmen asli.
            # 4. Reset akumulator untuk kalimat berikutnya.

            logical_sentences: List[Dict] = []  # Daftar untuk menyimpan kalimat-kalimat logis yang terbentuk
            current_sentence_chunks_texts: List[str] = [] # Akumulator untuk teks dari segmen-segmen saat ini
            current_sentence_chunks_timestamps: List[Tuple[float, float]] = [] # Akumulator untuk timestamp segmen-segmen saat ini

            for idx, seg in enumerate(segments):
                current_sentence_chunks_texts.append(seg['text'])
                current_sentence_chunks_timestamps.append(seg['timestamp'])

                # Cek apakah segmen ini mengakhiri sebuah kalimat
                is_last_segment: bool = (idx == len(segments) - 1)
                # Periksa apakah teks segmen (setelah di-strip) diakhiri dengan tanda baca umum.
                ends_with_punctuation: bool = seg['text'].strip().endswith(('.', '?', '!'))

                # Jika akhir kalimat terdeteksi (karena tanda baca atau ini segmen terakhir) DAN ada teks yang terakumulasi:
                if (ends_with_punctuation or is_last_segment) and current_sentence_chunks_texts:
                    # Gabungkan semua teks dari chunk yang terakumulasi menjadi satu string kalimat.
                    full_sentence_text: str = " ".join(current_sentence_chunks_texts).strip()

                    # Normalisasi spasi:
                    # 1. Pastikan ada satu spasi setelah tanda baca akhir kalimat (jika diikuti teks lain).
                    #    Contoh: "Halo dunia.Ini Budi." -> "Halo dunia. Ini Budi."
                    full_sentence_text = re.sub(r'\s*([.?!])\s*', r'\1 ', full_sentence_text)
                    # 2. Gabungkan beberapa spasi menjadi satu spasi.
                    #    Contoh: "Ini    kalimat." -> "Ini kalimat."
                    full_sentence_text = re.sub(r'\s+', ' ', full_sentence_text).strip()

                    if full_sentence_text: # Hanya proses jika kalimat tidak kosong setelah normalisasi
                        # Timestamp awal adalah waktu mulai dari chunk pertama dalam kalimat ini.
                        start_time_s: float = current_sentence_chunks_timestamps[0][0]
                        # Timestamp akhir adalah waktu selesai dari chunk terakhir dalam kalimat ini.
                        end_time_s: float = current_sentence_chunks_timestamps[-1][1]

                        logical_sentences.append({
                            'text': full_sentence_text, # Teks kalimat lengkap
                            'timestamp': [start_time_s, end_time_s], # [mulai_detik, selesai_detik]
                            'original_chunks_count': len(current_sentence_chunks_texts) # Jumlah segmen Whisper asli
                        })

                    # Reset akumulator untuk mempersiapkan kalimat logis berikutnya.
                    current_sentence_chunks_texts = []
                    current_sentence_chunks_timestamps = []
            
            # Tulis kalimat-kalimat yang sudah diterjemahkan ke dalam file SRT.
            logging.info(f"Writing {len(logical_sentences)} logical sentences to SRT file: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f: # Pastikan encoding UTF-8
                for i, sentence_data in enumerate(logical_sentences):
                    translated_text: str = self._translate_sentence(sentence_data['text'])
                    start_s, end_s = sentence_data['timestamp']
                    # Format timestamps for SRT
                    start_time_str = str(datetime.timedelta(seconds=start_s)).split('.')[0] + f",{int((start_s % 1) * 1000):03}"
                    end_time_str = str(datetime.timedelta(seconds=end_s)).split('.')[0] + f",{int((end_s % 1) * 1000):03}"
                    f.write(f"{i+1}\n{start_time_str} --> {end_time_str}\n{translated_text}\n\n")

# --- Bagian 3: Kelas-Kelas Pemoles Profesional (Tahap 2) ---

@dataclass
class SubtitleStandards:
    """
    Menyimpan standar dan batasan untuk pemformatan subtitle.
    """
    MAX_LINES: int = 2  # Jumlah baris maksimum per subtitle.
    MAX_CHARS_PER_LINE: int = 42  # Jumlah karakter maksimum per baris.
    MIN_READING_SPEED: float = 15.0  # Kecepatan membaca minimum (karakter per detik, tanpa spasi).
    MAX_READING_SPEED: float = 25.0  # Kecepatan membaca maksimum (karakter per detik, tanpa spasi).
    MIN_DURATION: float = 0.8  # Durasi minimum subtitle dalam detik.
    MAX_DURATION: float = 7.0  # Durasi maksimum subtitle dalam detik.
    MIN_GAP: float = 0.083  # Jarak minimum antar subtitle dalam detik (sekitar 2 frame pada 24fps).
    PREFERRED_READING_SPEED: float = 21.0 # Kecepatan membaca yang diutamakan untuk perhitungan durasi saat pemisahan.
    merge_gap_threshold: float = 0.75 # Batas jarak (detik) untuk menggabungkan subtitle yang berdekatan.

    @classmethod
    def from_config(cls, config_dict: Dict) -> 'SubtitleStandards':
        """
        Membuat instance SubtitleStandards dari dictionary konfigurasi.
        Args:
            config_dict (Dict): Dictionary yang mungkin berisi bagian 'standards'.
        Returns:
            SubtitleStandards: Instance dengan nilai default atau nilai dari config.
        """
        standards_config: Dict = config_dict.get('standards', {})
        instance = cls()
        for key, value in standards_config.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

class IntelligentLineBreaker:
    """
    Memecah teks menjadi beberapa baris subtitle dengan mempertimbangkan panjang maksimal,
    keseimbangan antar baris, dan aturan gramatikal dasar Bahasa Indonesia.
    """
    # Kumpulan konjungsi umum Bahasa Indonesia yang sebaiknya tidak mengawali baris kedua jika memungkinkan,
    # atau menandakan titik pemisahan yang baik jika baris kedua dimulai dengannya.
    INDONESIAN_CONJUNCTIONS: frozenset[str] = frozenset([
        'yang', 'dan', 'atau', 'tetapi', 'namun', 'sedangkan', 'melainkan', 'serta', 'lalu', 'kemudian',
        'jika', 'kalau', 'ketika', 'saat', 'sebelum', 'sesudah', 'karena', 'sebab', 'agar', 'supaya',
        'meskipun', 'walaupun'
    ])

    def __init__(self, standards: SubtitleStandards) -> None:
        """
        Inisialisasi IntelligentLineBreaker.
        Args:
            standards (SubtitleStandards): Objek standar subtitle yang akan digunakan.
        """
        self.standards: SubtitleStandards = standards
        logging.info("IntelligentLineBreaker initialized with enhanced grammatical and balancing rules.")

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Memotong teks jika melebihi panjang maksimal, menambahkan elipsis.
        Args:
            text (str): Teks input.
            max_length (int): Panjang karakter maksimal.
        Returns:
            str: Teks yang dipotong jika perlu.
        """
        if len(text) <= max_length:
            return text
        # Cari spasi terakhir sebelum max_length - 3 (untuk elipsis "...")
        pos: int = text.rfind(' ', 0, max_length - 3)
        if pos != -1:
            return text[:pos] + "..."
        else: # Jika tidak ada spasi, potong paksa
            return text[:max_length - 3] + "..."

    def break_lines(self, text: str) -> str:
        """
        Memecah teks input menjadi beberapa baris sesuai standar.
        Strategi:
        1. Tangani kasus-kasus sederhana (teks kosong, teks pendek).
        2. Lakukan pemotongan keseluruhan jika teks terlalu panjang untuk jumlah baris maksimum.
        3. Jika target 1 baris: potong teks agar pas.
        4. Jika target 2 baris: gunakan _break_into_two_lines untuk pemecahan optimal.
        5. Jika target > 2 baris: gunakan pendekatan greedy, lalu potong baris terakhir jika perlu.
        Args:
            text (str): Teks yang akan dipecah.
        Returns:
            str: Teks yang telah dipecah menjadi beberapa baris (dipisahkan newline).
        """
        max_chars: int = self.standards.MAX_CHARS_PER_LINE
        max_lines: int = self.standards.MAX_LINES

        if not text or not text.strip():
            return ""
        if len(text) <= max_chars: # Cukup untuk satu baris
            return text

        # Pemotongan keseluruhan jika teks terlalu panjang bahkan untuk max_lines
        if len(text) > max_chars * max_lines:
            text = self._truncate_text(text, max_chars * max_lines)

        words: List[str] = text.split()
        if not words:
            return ""

        if max_lines == 1:
            # Teks sudah pasti > max_chars dari pengecekan di atas.
            return self._truncate_text(" ".join(words), max_chars)

        if max_lines == 2:
            return self._break_into_two_lines(words, max_chars, self.INDONESIAN_CONJUNCTIONS)

        # Logika untuk max_lines > 2 (Pendekatan Greedy)
        # Mengisi baris satu per satu hingga batas karakter atau hingga baris sebelum terakhir.
        # Baris terakhir akan menampung sisa kata, lalu dipotong jika perlu.
        lines_output: List[str] = []
        current_line_words: List[str] = []
        current_length: int = -1  # -1 untuk menangani spasi di awal join

        for word in words:
            word_len: int = len(word)
            if not current_line_words: # Kata pertama di baris baru
                current_line_words.append(word)
                current_length = word_len
            elif current_length + 1 + word_len <= max_chars: # Tambahkan kata jika muat
                current_line_words.append(word)
                current_length += 1 + word_len
            else: # Kata tidak muat, finalisasi baris saat ini
                lines_output.append(" ".join(current_line_words))
                current_line_words = [word] # Mulai baris baru dengan kata saat ini
                current_length = word_len
                if len(lines_output) == max_lines - 1: # Jika sudah mengisi semua baris kecuali baris terakhir
                    # Sisa kata akan masuk ke current_line_words dan ditambahkan setelah loop.
                    break

        if current_line_words: # Tambahkan sisa baris terakhir
            lines_output.append(" ".join(current_line_words))

        # Pastikan tidak melebihi max_lines (jika teks sangat pendek dan banyak kata kecil)
        final_lines: List[str] = lines_output[:max_lines]

        # Potong baris terakhir dari hasil final jika masih terlalu panjang
        if final_lines and len(final_lines[-1]) > max_chars:
            final_lines[-1] = self._truncate_text(final_lines[-1], max_chars)

        return "\n".join(final_lines)

    def _break_into_two_lines(self, words: List[str], max_chars: int, conjunctions: frozenset[str]) -> str:
        """
        Memecah daftar kata menjadi dua baris secara optimal berdasarkan sistem penalti/bonus.
        Tujuannya adalah mencari titik pemisahan yang:
        - Tidak melebihi panjang karakter per baris.
        - Menghindari "orphan" (satu kata pendek) di baris kedua.
        - Menyeimbangkan panjang antar baris.
        - Memberi bonus jika pemisahan terjadi setelah koma atau jika baris kedua dimulai dengan konjungsi.
        Args:
            words (List[str]): Daftar kata yang akan dipecah.
            max_chars (int): Panjang karakter maksimum per baris.
            conjunctions (frozenset[str]): Kumpulan konjungsi untuk panduan gramatikal.
        Returns:
            str: String dua baris yang dipecah (dipisahkan newline), atau satu baris jika tidak bisa dipecah dengan baik.
        """
        n_words: int = len(words)
        if n_words == 0:
            return ""
        if n_words == 1: # Kasus patologis: satu kata yang sangat panjang
            return self._truncate_text(words[0], max_chars)

        # Inisialisasi best_split: defaultnya semua kata di baris pertama.
        # Ini akan digunakan jika tidak ada pemisahan yang "baik" ditemukan.
        best_split: Dict[str, any] = {
            'penalty': float('inf'),
            'line1': " ".join(words),
            'line2': ""
        }

        # Iterasi melalui semua kemungkinan titik pemisahan (k adalah jumlah kata di baris pertama)
        for k in range(1, n_words):
            line1_words: List[str] = words[:k]
            line2_words: List[str] = words[k:]

            line1: str = " ".join(line1_words)
            # Baris kedua pasti memiliki kata karena k berjalan hingga n_words-1
            line2: str = " ".join(line2_words)

            penalty: float = 0

            # 1. Penalti Panjang Baris
            if len(line1) > max_chars:
                penalty += 1000 + (len(line1) - max_chars) * 10 # Penalti besar + per karakter
            if len(line2) > max_chars: # Harus periksa line2 juga
                penalty += 1000 + (len(line2) - max_chars) * 10

            # 2. Penalti "Orphan" di Baris Kedua
            # Jika baris kedua hanya satu kata dan kata itu pendek, beri penalti.
            if len(line2_words) == 1 and len(line1_words) > 0 and len(line2) <= int(max_chars * 0.33):
                penalty += 50

            # 3. Penalti Ketidakseimbangan Panjang (hanya jika kedua baris valid)
            if len(line1) <= max_chars and len(line2) <= max_chars:
                penalty += abs(len(line1) - len(line2))

            # 4. Bonus Gramatikal (mengurangi penalti)
            if line1_words and line1_words[-1].endswith(','): # Lebih baik memecah setelah koma
                penalty -= 20
            if line2_words and line2_words[0].lower() in conjunctions: # Lebih baik jika baris kedua dimulai konjungsi
                penalty -= 20

            # Update pemisahan terbaik jika penalti saat ini lebih rendah
            if penalty < best_split['penalty']:
                best_split = {'penalty': penalty, 'line1': line1, 'line2': line2}

            # Catatan: Loop ini secara inheren mencoba menempatkan lebih banyak kata di baris pertama pada awalnya.
            # Jika sebuah pemisahan valid (misalnya, line1 dan line2 <= max_chars), penalti akan relatif rendah,
            # dan bisa menjadi best_split. Jika teksnya pendek, " ".join(words) di awal mungkin tetap jadi best_split
            # jika semua upaya pemisahan menghasilkan penalti tinggi (misalnya, karena line2 menjadi terlalu panjang).

        final_l1: str = best_split['line1']
        final_l2: str = best_split['line2']

        # Pemotongan Final (Jaring Pengaman) - jika sistem penalti gagal mencegah overflow
        if len(final_l1) > max_chars:
            final_l1 = self._truncate_text(final_l1, max_chars)

        if final_l2 and len(final_l2) > max_chars: # Cek final_l2 ada sebelum cek panjangnya
            final_l2 = self._truncate_text(final_l2, max_chars)

        if not final_l2.strip(): # Jika baris kedua kosong atau hanya spasi
            return final_l1

        return f"{final_l1}\n{final_l2}"

class ProfessionalSubtitleProcessor:
    """
    Memproses subtitle mentah menjadi format profesional dengan menerapkan
    aturan standar industri terkait pemecahan baris, durasi, dan kecepatan membaca.
    """
    def __init__(self, config: Dict) -> None:
        """
        Inisialisasi ProfessionalSubtitleProcessor.
        Args:
            config (Dict): Dictionary konfigurasi, biasanya dari file YAML.
        """
        self.config: Dict = config or {}
        self.standards: SubtitleStandards = SubtitleStandards.from_config(self.config)
        self.line_breaker: IntelligentLineBreaker = IntelligentLineBreaker(self.standards)
        logging.info("ProfessionalSubtitleProcessor initialized.")

    def process_from_file(self, input_path: str) -> List[Dict]:
        """
        Memproses file SRT input dan menghasilkan daftar subtitle yang diformat secara profesional.
        Alur Pemrosesan:
        1. Parse SRT mentah.
        2. Gabungkan subtitle berurutan yang memenuhi syarat (pendek, tidak diakhiri tanda baca, dll.).
        3. Terapkan aturan timing (durasi min/max, CPS min/max) dan pisahkan subtitle jika perlu.
        4. Pecah teks setiap subtitle menjadi beberapa baris sesuai standar.
        Args:
            input_path (str): Path ke file SRT input.
        Returns:
            List[Dict]: Daftar subtitle yang telah diproses, masing-masing berupa dictionary
                        dengan 'start', 'end', dan 'text'.
        """
        logging.info(f"Starting professional processing for: {input_path}")
        raw_subs: List[Dict] = self._parse_srt(input_path)
        if not raw_subs:
            logging.warning("No subtitles parsed from input file.")
            return []
        logging.info(f"Parsed {len(raw_subs)} raw subtitles.")

        # 1. Gabungkan subtitle yang berdekatan dan memenuhi syarat
        merged_subs: List[Dict] = self._merge_sequential_subtitles(raw_subs)
        if not merged_subs:
            logging.warning("No subtitles after merging step.")
            return []
        logging.info(f"Reduced to {len(merged_subs)} subtitles after merging.")

        # 2. Terapkan aturan timing dan pisahkan subtitle jika perlu
        timed_adjusted_subs: List[Dict] = self._apply_timing_rules_and_split(merged_subs)
        if not timed_adjusted_subs:
            logging.warning("No subtitles after timing adjustment and splitting step.")
            return []
        logging.info(f"Adjusted to {len(timed_adjusted_subs)} subtitles after timing rules.")

        # 3. Pecah teks menjadi baris-baris yang sesuai standar
        fully_processed_subs: List[Dict] = []
        for sub_data in timed_adjusted_subs:
            # Pastikan 'text' ada dan berupa string sebelum dipecah
            text_to_break: str = sub_data.get('text', '')
            if not isinstance(text_to_break, str):
                logging.warning(f"Subtitle text is not a string: {text_to_break}. Skipping line breaking for this sub.")
                formatted_text = str(text_to_break) # Atau kosongkan
            else:
                formatted_text = self.line_breaker.break_lines(text_to_break)

            fully_processed_subs.append({
                'start': sub_data['start'],
                'end': sub_data['end'],
                'text': formatted_text
            })
        logging.info(f"Finalized {len(fully_processed_subs)} subtitles after line breaking.")

        return fully_processed_subs

    def _parse_srt(self, file_path: str) -> List[Dict]:
        """
        Membaca file SRT dan mengonversinya menjadi daftar dictionary.
        Args:
            file_path (str): Path ke file SRT.
        Returns:
            List[Dict]: Daftar subtitle, masing-masing dengan 'start', 'end', 'text'.
        """
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logging.warning(f"SRT file not found or empty: {file_path}")
            return []
        try:
            subs = pysrt.open(file_path, encoding='utf-8')
            return [{
                'start': (s.start.hours * 3600 + s.start.minutes * 60 + s.start.seconds + s.start.milliseconds / 1000.0),
                'end': (s.end.hours * 3600 + s.end.minutes * 60 + s.end.seconds + s.end.milliseconds / 1000.0),
                'text': s.text
            } for s in subs]
        except Exception as e:
            logging.error(f"Error parsing SRT file {file_path}: {e}", exc_info=True)
            return []


    def _merge_sequential_subtitles(self, subtitles: List[Dict]) -> List[Dict]:
        """
        Menggabungkan subtitle berurutan jika memenuhi kriteria tertentu (misalnya, gap pendek,
        tidak ada tanda baca akhir di subtitle pertama, subtitle kedua tidak diawali huruf kapital).
        Args:
            subtitles (List[Dict]): Daftar subtitle input.
        Returns:
            List[Dict]: Daftar subtitle setelah digabungkan.
        """
        if not subtitles:
            return []

        merged_subs: List[Dict] = []
        # Tidak perlu cek 'if not subtitles:' lagi karena sudah di atas.
        # Langsung copy subtitle pertama untuk memulai.
        current_sub: Dict = subtitles[0].copy()

        for i in range(1, len(subtitles)):
            next_sub: Dict = subtitles[i]

            gap: float = next_sub['start'] - current_sub['end']
            # Gabungkan teks untuk pemeriksaan panjang, tambahkan spasi di antaranya.
            prospective_combined_text: str = current_sub['text'] + " " + next_sub['text']
            combined_text_len: int = len(prospective_combined_text) # Hitung panjang teks gabungan
            combined_duration: float = next_sub['end'] - current_sub['start']

            # Kondisi untuk penggabungan:
            # 1. Jarak (gap) antar subtitle cukup kecil.
            # 2. Teks subtitle saat ini tidak diakhiri dengan tanda baca titik, tanya, atau seru.
            # 3. Teks subtitle berikutnya tidak dimulai dengan huruf kapital (heuristik untuk awal kalimat baru).
            # 4. Panjang teks gabungan tidak melebihi batas yang diizinkan (dengan sedikit buffer).
            # 5. Durasi gabungan tidak melebihi durasi maksimum subtitle.
            can_merge: bool = (
                gap < self.standards.merge_gap_threshold and
                not current_sub['text'].strip().endswith(('.', '!', '?')) and
                not (next_sub['text'] and next_sub['text'][0].isupper()) and
                combined_text_len < (self.standards.MAX_CHARS_PER_LINE * self.standards.MAX_LINES) * 0.95 and # 95% buffer
                combined_duration < self.standards.MAX_DURATION
            )

            if can_merge:
                # Lakukan penggabungan
                current_sub['text'] = prospective_combined_text # Gunakan teks yang sudah digabung
                current_sub['end'] = next_sub['end'] # Perbarui waktu akhir
            else:
                # Tidak bisa digabung, simpan subtitle saat ini dan jadikan subtitle berikutnya sebagai 'current'
                merged_subs.append(current_sub)
                current_sub = next_sub.copy()

        merged_subs.append(current_sub) # Tambahkan subtitle terakhir yang diproses
        return merged_subs

    def _split_subtitle(self, sub: Dict, approx_char_split_idx: int) -> List[Dict]:
        """
        Memisahkan satu subtitle menjadi dua berdasarkan perkiraan indeks karakter.
        Durasi dan timestamp untuk subtitle baru dihitung berdasarkan kecepatan membaca yang diutamakan.
        Args:
            sub (Dict): Subtitle yang akan dipisah.
            approx_char_split_idx (int): Perkiraan indeks karakter untuk memisahkan.
        Returns:
            List[Dict]: Daftar berisi dua subtitle hasil pemisahan, atau satu subtitle asli jika pemisahan gagal.
        """
        text: str = sub['text']
        if approx_char_split_idx <= 0 or approx_char_split_idx >= len(text):
            logging.debug(f"Cannot split subtitle: approx_char_split_idx ({approx_char_split_idx}) is out of bounds for text length {len(text)}.")
            return [sub] # Tidak bisa dipisah secara berarti

        # Cari titik pemisahan aktual berdasarkan spasi terdekat.
        # Prioritas: spasi sebelum indeks, lalu spasi setelah indeks, terakhir potong paksa.
        break_pos: int = text.rfind(' ', 0, approx_char_split_idx) # Cari spasi sebelum atau di approx_char_split_idx
        if break_pos == -1: # Jika tidak ada spasi sebelum, coba cari setelah
            break_pos = text.find(' ', approx_char_split_idx)
        if break_pos == -1: # Jika masih tidak ada spasi, gunakan indeks perkiraan (potong paksa)
            logging.debug(f"No suitable space found for splitting near {approx_char_split_idx}. Using hard split.")
            break_pos = approx_char_split_idx

        text1: str = text[:break_pos].strip()
        text2: str = text[break_pos:].strip()

        if not text1 or not text2: # Jika salah satu bagian kosong setelah dipisah
            logging.debug("Splitting resulted in an empty part. Returning original subtitle.")
            return [sub]

        # Fungsi helper untuk menghitung panjang teks tanpa spasi (untuk CPS)
        len_no_space = lambda t: len(re.sub(r'\s', '', t))

        # Hitung durasi untuk bagian pertama
        # Durasi minimal adalah standar MIN_DURATION, atau berdasarkan PREFERRED_READING_SPEED.
        dur1: float = max(len_no_space(text1) / self.standards.PREFERRED_READING_SPEED, self.standards.MIN_DURATION)
        end1: float = sub['start'] + dur1
        sub1: Dict = {'text': text1, 'start': sub['start'], 'end': end1}

        # Hitung durasi untuk bagian kedua
        start2: float = end1 + self.standards.MIN_GAP # Mulai setelah gap minimum
        dur2: float = max(len_no_space(text2) / self.standards.PREFERRED_READING_SPEED, self.standards.MIN_DURATION)
        end2: float = start2 + dur2
        # Catatan: end2 mungkin perlu penyesuaian lebih lanjut oleh _apply_timing_rules_and_split
        # jika bertabrakan dengan subtitle berikutnya atau melebihi MAX_DURATION secara keseluruhan.
        sub2: Dict = {'text': text2, 'start': start2, 'end': end2}

        logging.debug(f"Split subtitle into two: '{text1}' ({sub1['start']:.3f}-{sub1['end']:.3f}) and '{text2}' ({sub2['start']:.3f}-{sub2['end']:.3f})")
        return [sub1, sub2]

    def _apply_timing_rules_and_split(self, subs_to_process: List[Dict]) -> List[Dict]:
        """
        Menerapkan aturan timing (durasi min/max, CPS min/max) secara iteratif ke daftar subtitle.
        Subtitle akan dipecah jika aturan dilanggar dan tidak bisa disesuaikan dengan mengubah waktu akhir.
        Args:
            subs_to_process (List[Dict]): Daftar subtitle yang akan diproses.
        Returns:
            List[Dict]: Daftar subtitle setelah aturan timing diterapkan dan pemisahan dilakukan.
        """
        final_subs: List[Dict] = []
        if not subs_to_process:
            return final_subs

        working_list: List[Dict] = list(subs_to_process) # Salinan yang bisa dimodifikasi
        idx: int = 0 # Indeks untuk iterasi pada working_list

        # MAX_ITERATIONS_PER_SUB: Batas keamanan untuk mencegah loop tak terbatas jika aturan saling bertentangan
        # atau menyebabkan osilasi dalam penyesuaian subtitle.
        MAX_ITERATIONS_PER_SUB: int = 5

        # Fungsi helper untuk menghitung panjang teks tanpa spasi (untuk CPS)
        len_no_space_func = lambda t: len(re.sub(r'\s', '', t))

        while idx < len(working_list):
            sub: Dict = working_list[idx].copy() # Ambil salinan subtitle yang sedang diproses
            iteration_count: int = 0
            # processed_in_iteration: Flag untuk menandai jika subtitle saat ini dipecah.
            # Jika True, loop luar akan mengevaluasi ulang bagian pertama dari subtitle yang baru dipecah (di indeks `idx` yang sama).
            processed_in_iteration: bool = False

            # Loop internal untuk menerapkan aturan secara berulang pada satu subtitle
            # hingga tidak ada perubahan atau batas iterasi tercapai.
            while iteration_count < MAX_ITERATIONS_PER_SUB:
                iteration_count += 1
                original_sub_state: Dict = sub.copy() # Simpan kondisi awal untuk deteksi perubahan

                duration: float = sub['end'] - sub['start']
                text_len_no_space: int = len_no_space_func(sub['text'])

                # --- Aturan 1: Durasi Maksimum ---
                if duration > self.standards.MAX_DURATION:
                    logging.debug(f"Sub {idx} violates MAX_DURATION ({duration:.2f}s > {self.standards.MAX_DURATION:.2f}s). Attempting split.")
                    # Hitung indeks pemisahan secara proporsional berdasarkan MAX_DURATION.
                    split_idx_prop: int = int(len(sub['text']) * (self.standards.MAX_DURATION / duration))
                    split_subs: List[Dict] = self._split_subtitle(sub, split_idx_prop)
                    if len(split_subs) == 2:
                        # Ganti subtitle saat ini dengan dua subtitle hasil pemisahan.
                        working_list[idx:idx+1] = split_subs
                        processed_in_iteration = True; break # Keluar loop internal, proses ulang dari idx
                if processed_in_iteration: continue # Lanjut ke iterasi berikutnya dari loop luar (idx tetap)

                # --- Aturan 2: Durasi Minimum ---
                duration = sub['end'] - sub['start'] # Hitung ulang durasi jika ada perubahan
                if duration < self.standards.MIN_DURATION:
                    logging.debug(f"Sub {idx} violates MIN_DURATION ({duration:.2f}s < {self.standards.MIN_DURATION:.2f}s). Adjusting end time.")
                    potential_end: float = sub['start'] + self.standards.MIN_DURATION
                    # Pastikan penyesuaian tidak melebihi MAX_DURATION atau bertabrakan dengan subtitle berikutnya.
                    next_sub_start_time: float = working_list[idx+1]['start'] if (idx + 1) < len(working_list) else float('inf')
                    sub['end'] = min(potential_end,
                                     sub['start'] + self.standards.MAX_DURATION,
                                     next_sub_start_time - self.standards.MIN_GAP if next_sub_start_time != float('inf') else float('inf'))

                # --- Aturan 3: Kecepatan Membaca Maksimum (Max CPS) ---
                duration = sub['end'] - sub['start'] # Hitung ulang
                text_len_no_space = len_no_space_func(sub['text']) # Hitung ulang
                current_cps: float = text_len_no_space / duration if duration > 0 else float('inf')

                if current_cps > self.standards.MAX_READING_SPEED:
                    logging.debug(f"Sub {idx} violates MAX_READING_SPEED ({current_cps:.2f} CPS > {self.standards.MAX_READING_SPEED:.2f} CPS).")
                    required_duration: float = text_len_no_space / self.standards.MAX_READING_SPEED
                    potential_end: float = sub['start'] + required_duration
                    next_sub_start_time: float = working_list[idx+1]['start'] if (idx + 1) < len(working_list) else float('inf')

                    # Coba perpanjang durasi jika memungkinkan
                    if potential_end <= sub['start'] + self.standards.MAX_DURATION and \
                       potential_end <= (next_sub_start_time - self.standards.MIN_GAP if next_sub_start_time != float('inf') else float('inf')):
                        logging.debug(f"  Adjusting end time for sub {idx} to meet MAX_READING_SPEED.")
                        sub['end'] = potential_end
                    else: # Jika tidak bisa diperpanjang, harus dipecah
                        logging.debug(f"  Cannot adjust end time for sub {idx}. Attempting split.")
                        # Pemisahan sederhana menjadi dua bagian (bisa lebih canggih)
                        split_idx_half: int = int(len(sub['text']) * 0.5)
                        split_subs = self._split_subtitle(sub, split_idx_half)
                        if len(split_subs) == 2:
                            working_list[idx:idx+1] = split_subs
                            processed_in_iteration = True; break
                if processed_in_iteration: continue

                # --- Aturan 4: Kecepatan Membaca Minimum (Min CPS) ---
                # Ini lebih bersifat opsional; penyesuaian hanya jika tidak melanggar MIN_DURATION.
                duration = sub['end'] - sub['start'] # Hitung ulang
                text_len_no_space = len_no_space_func(sub['text']) # Hitung ulang
                current_cps = text_len_no_space / duration if duration > 0 else 0

                # Hanya perpendek jika durasi saat ini sedikit lebih panjang dari MIN_DURATION,
                # untuk memberi prioritas pada MIN_DURATION.
                if current_cps < self.standards.MIN_READING_SPEED and duration > (self.standards.MIN_DURATION * 1.05) :
                    logging.debug(f"Sub {idx} violates MIN_READING_SPEED ({current_cps:.2f} CPS < {self.standards.MIN_READING_SPEED:.2f} CPS). Adjusting end time.")
                    target_duration: float = max(text_len_no_space / self.standards.MIN_READING_SPEED, self.standards.MIN_DURATION)
                    if sub['start'] + target_duration < sub['end']: # Pastikan benar-benar memperpendek
                         sub['end'] = sub['start'] + target_duration

                # Jika tidak ada perubahan pada subtitle dalam iterasi ini, hentikan loop internal.
                if sub == original_sub_state:
                    logging.debug(f"No changes for sub {idx} in iteration {iteration_count}. Moving to next sub or finalization.")
                    break

            if not processed_in_iteration: # Jika subtitle tidak dipecah (meskipun mungkin dimodifikasi)
                final_subs.append(sub) # Tambahkan subtitle yang sudah diproses (atau tidak berubah) ke hasil akhir
                idx += 1 # Lanjut ke subtitle berikutnya di working_list
            # Jika processed_in_iteration True, idx TIDAK bertambah. Loop luar akan mengulang
            # dari indeks saat ini, yang sekarang berisi bagian pertama dari subtitle yang baru dipecah.

        return final_subs

    def write_srt_file(self, subtitles: List[Dict], file_path: str):
        """
        Menulis daftar subtitle ke dalam file SRT.
        Args:
            subtitles (List[Dict]): Daftar subtitle yang akan ditulis.
            file_path (str): Path untuk menyimpan file SRT.
        """
        logging.info(f"Writing {len(subtitles)} processed subtitles to {file_path}")
        subs = pysrt.SubRipFile()
        for i, sub_data in enumerate(subtitles, 1):
            item = pysrt.SubRipItem(index=i, text=sub_data['text'])
            start_seconds: float = sub_data['start']
            end_seconds: float = sub_data['end']

            # Konversi detik ke format waktu SubRipTime
            s_h, s_rem = divmod(start_seconds, 3600)
            s_m, s_s_float = divmod(s_rem, 60)
            s_s = int(s_s_float)
            s_ms = int((s_s_float % 1) * 1000)
            item.start.hours = int(s_h); item.start.minutes = int(s_m); item.start.seconds = s_s; item.start.milliseconds = s_ms

            e_h, e_rem = divmod(end_seconds, 3600)
            e_m, e_s_float = divmod(e_rem, 60)
            e_s = int(e_s_float)
            e_ms = int((e_s_float % 1) * 1000)
            item.end.hours = int(e_h); item.end.minutes = int(e_m); item.end.seconds = e_s; item.end.milliseconds = e_ms

            subs.append(item)
        try:
            subs.save(file_path, encoding='utf-8')
            logging.info(f"SRT file saved successfully to {file_path}")
        except Exception as e:
            logging.error(f"Error saving SRT file {file_path}: {e}", exc_info=True)

# --- Bagian 4: Eksekusi Utama ---
            item = pysrt.SubRipItem(index=i, text=sub_data['text'])
            start_seconds = sub_data['start']; end_seconds = sub_data['end']
            s_h, s_rem = divmod(start_seconds, 3600); s_m, s_s = divmod(s_rem, 60)
            item.start.hours = int(s_h); item.start.minutes = int(s_m); item.start.seconds = int(s_s); item.start.milliseconds = int((start_seconds % 1) * 1000)
            e_h, e_rem = divmod(end_seconds, 3600); e_m, e_s = divmod(e_rem, 60)
            item.end.hours = int(e_h); item.end.minutes = int(e_m); item.end.seconds = int(e_s); item.end.milliseconds = int((end_seconds % 1) * 1000)
            subs.append(item)
        subs.save(file_path, encoding='utf-8')

# --- Bagian 4: Eksekusi Utama ---

def main() -> None:
    """
    Fungsi utama untuk menjalankan seluruh alur kerja pembuatan dan pemolesan subtitle.
    Alur kerja dibagi menjadi dua tahap utama:
    1. Generasi Konten Mentah: Transkripsi audio dari video dan terjemahan awal ke Bahasa Indonesia.
       Hasilnya adalah file SRT mentah.
    2. Pemolesan Profesional: Menerapkan berbagai aturan standar subtitle (pemecahan baris,
       durasi, kecepatan membaca, penggabungan) pada file SRT mentah untuk menghasilkan
       output akhir yang berkualitas tinggi.
    """
    # Konfigurasi logging dasar untuk output informasi proses.
    # 'force=True' digunakan jika logging mungkin sudah dikonfigurasi oleh library lain (misalnya di Kaggle).
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - (%(module)s:%(lineno)d) - %(message)s',
        force=True
    )
    
    # --- TAHAP 1: GENERASI KONTEN MENTAH ---
    logging.info("--- Memulai TAHAP 1: Generasi Konten Mentah ---")
    try:
        content_generator: ContentGenerator = ContentGenerator()
        content_generator.run() # Menjalankan transkripsi dan terjemahan awal
        logging.info("✅ TAHAP 1 SELESAI. File 'raw_subtitle.srt' telah dibuat.")
    except Exception as e:
        # Tangani error kritis yang mungkin terjadi selama generasi konten mentah.
        logging.critical(f"❌ TAHAP 1 GAGAL KRITIS. Error: {e}", exc_info=True)
        return # Hentikan eksekusi jika tahap 1 gagal

    # --- TAHAP 2: PEMOLESAN PROFESIONAL ---
    logging.info("\n--- Memulai TAHAP 2: Pemolesan Profesional ---")
    
    # Path file input (hasil dari Tahap 1) dan output untuk Tahap 2.
    raw_input_srt: str = ContentConfig.RAW_SRT_OUTPUT_PATH # Menggunakan path dari ContentConfig
    final_output_srt: str = "/kaggle/working/FINAL_professional_subtitle.srt" # Path output akhir

    # Path ke file konfigurasi YAML yang berisi standar pemolesan.
    # Diasumsikan file ini ada di dataset input.
    config_file_path: str = os.path.join(ContentConfig.DATASET_PATH, "config.yaml")

    # Pastikan file SRT mentah (input untuk tahap ini) ada.
    if not os.path.exists(raw_input_srt) or os.path.getsize(raw_input_srt) == 0:
        logging.critical(f"❌ TAHAP 2 GAGAL: File input '{raw_input_srt}' tidak ditemukan atau kosong.")
        return # Hentikan jika input tidak valid

    try:
        # Muat konfigurasi pemolesan dari file YAML.
        config: Dict
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Inisialisasi prosesor subtitle profesional dengan konfigurasi yang dimuat.
        processor: ProfessionalSubtitleProcessor = ProfessionalSubtitleProcessor(config)

        # Proses file SRT mentah untuk menerapkan semua aturan pemolesan.
        final_subtitles: List[Dict] = processor.process_from_file(raw_input_srt)

        # Tulis hasil subtitle yang sudah dipoles ke file SRT akhir.
        processor.write_srt_file(final_subtitles, final_output_srt)

        logging.info(f"✅ TAHAP 2 SELESAI. File subtitle profesional disimpan di {final_output_srt}")
    except Exception as e:
        # Tangani error kritis yang mungkin terjadi selama pemolesan.
        logging.critical(f"❌ TAHAP 2 GAGAL KRITIS. Error: {e}", exc_info=True)

if __name__ == "__main__":
    # Panggil fungsi main jika script dijalankan secara langsung.
    main()