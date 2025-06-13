# =======================================================================
# AVT SUBTITLER PRO - FINAL ALL-IN-ONE SCRIPT (No spaCy)
# Version: 6.0.1 (Final Logic and Structure)
# =======================================================================

# --- Bagian 1: Impor & Konfigurasi Awal ---
import os, sys, json, re, time, logging, datetime, shutil
from typing import List, Dict, Tuple, Optional, frozenset
# from dataclasses import dataclass # Digantikan oleh Pydantic model
import torch
import moviepy.editor as mp
# pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor are removed as AudioTranscriber now uses faster-whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kaggle_secrets import UserSecretsClient
import pysrt
import yaml # Untuk memuat config.yaml
import argparse

# Impor Pydantic model dari config.py di root directory
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Menambahkan root ke sys.path
from config import AppConfigModel, ContentGenerationConfigModel, SubtitleStandardsModel, PathsConfigModel, DiarizationConfigModel, CheckpointingConfigModel

# Imports for Diarization
from pyannote.audio import Pipeline as PyannotePipeline
import torchaudio # For audio loading/info if needed by pyannote or for duration

# Imports for Faster Whisper
from faster_whisper import WhisperModel
# moviepy.editor as mp is already imported
# torch is already imported for torch.device
# `transformers.pipeline` might be removed if no longer used elsewhere, but let's keep it for now
# as other parts of the original script might have used it, or it might be reintroduced.
# For now, AudioTranscriber will not use it.

# --- Helper Function for Diarization ---
def get_speaker_for_timestamp(time_sec: float, diar_segments: List[Dict], default_speaker: str = "SPEAKER_00") -> str:
    """
    Determines the speaker for a given timestamp based on diarization segments.
    Args:
        time_sec (float): The timestamp (in seconds) to find the speaker for.
        diar_segments (List[Dict]): A list of speaker segments, each with 'start', 'end', and 'speaker'.
                                    Assumes segments are sorted by start time.
        default_speaker (str): The speaker ID to return if no segment matches.
    Returns:
        str: The identified speaker ID.
    """
    if not diar_segments:
        return default_speaker

    # Check for direct overlap
    for seg in diar_segments:
        if seg['start'] <= time_sec < seg['end']:
            return seg['speaker']

    # Handle cases where time_sec is outside any segment (e.g., before the first or after the last)
    if time_sec < diar_segments[0]['start']:
        # logging.debug(f"Timestamp {time_sec:.2f}s is before the first speaker segment. Assigning to first speaker: {diar_segments[0]['speaker']}.")
        return diar_segments[0]['speaker'] # Or default_speaker based on desired behavior

    if time_sec >= diar_segments[-1]['end']:
        # logging.debug(f"Timestamp {time_sec:.2f}s is after the last speaker segment. Assigning to last speaker: {diar_segments[-1]['speaker']}.")
        return diar_segments[-1]['speaker'] # Or default_speaker

    # Handle gaps between segments: assign to the speaker of the preceding segment
    for i in range(len(diar_segments) - 1):
        current_seg_end = diar_segments[i]['end']
        next_seg_start = diar_segments[i+1]['start']
        if current_seg_end <= time_sec < next_seg_start:
            # logging.debug(f"Timestamp {time_sec:.2f}s is in a gap. Assigning to previous speaker: {diar_segments[i]['speaker']}.")
            return diar_segments[i]['speaker']

    # Fallback if no other condition met (should ideally be covered by above checks)
    # logging.debug(f"Timestamp {time_sec:.2f}s did not fall into any specific segment or gap logic. Using default: {default_speaker}.")
    return default_speaker


# --- Fungsi Pemuatan Konfigurasi ---
def load_app_config(config_path: str = "config.yaml") -> AppConfigModel:
    """
    Memuat konfigurasi aplikasi dari file YAML dan memvalidasinya dengan Pydantic model.
    Args:
        config_path (str): Path ke file konfigurasi YAML.
    Returns:
        AppConfigModel: Instance Pydantic model AppConfigModel yang berisi konfigurasi.
    """
    logging.info(f"Mencoba memuat konfigurasi dari: {config_path}")
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not config_data: # Jika file kosong atau tidak valid YAML
            logging.warning(f"File konfigurasi {config_path} kosong atau bukan YAML valid. Menggunakan default.")
            config_data = {}
    else:
        logging.warning(f"File konfigurasi {config_path} tidak ditemukan. Menggunakan default.")

    try:
        app_config = AppConfigModel(**config_data)
        logging.info("Konfigurasi berhasil dimuat dan divalidasi.")

        # Auto-deteksi device jika tidak dispesifikasikan di config.yaml
        if app_config.content_generation.device is None:
            app_config.content_generation.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Device auto-detected and set to: {app_config.content_generation.device}")

        return app_config
    except Exception as e: # Lebih spesifik menangkap ValidationError dari Pydantic jika perlu
        logging.error(f"Error saat memuat atau memvalidasi konfigurasi dari {config_path}: {e}", exc_info=True)
        raise # Re-raise error setelah logging, karena konfigurasi penting


# --- Bagian 2: Kelas-Kelas Generator Konten (Tahap 1) ---

# Kelas ContentConfig lama DIHAPUS. Digantikan oleh AppConfigModel.paths dan AppConfigModel.content_generation.

class ContentGenerator:
    """
    Mengelola proses pembuatan konten mentah: transkripsi audio dan terjemahan awal.
    """
    def __init__(self, app_config: AppConfigModel) -> None: # Menggunakan AppConfigModel
        """
        Inisialisasi ContentGenerator dengan konfigurasi Pydantic.
        Args:
            app_config (AppConfigModel): Konfigurasi aplikasi Pydantic.
        """
        self.app_config: AppConfigModel = app_config
        self.cfg_content_gen: ContentGenerationConfigModel = app_config.content_generation
        self.paths_config: PathsConfigModel = app_config.paths

        # Path penting yang akan sering digunakan
        self.video_input_path: str = self.cfg_content_gen.video_input_path # Bisa di-override oleh UI
        self.mapping_json_path: str = os.path.join(self.paths_config.dataset_path, self.paths_config.mapping_json_filename)
        self.raw_srt_output_path: str = os.path.join(self.paths_config.working_directory, self.paths_config.raw_srt_filename)
        self.log_file_path: str = os.path.join(self.paths_config.working_directory, self.paths_config.log_filename)
        # Device sudah di-set di load_app_config

        self.translation_engine: ContentGenerator.TranslationEngine = self.TranslationEngine(app_config)
        self.transcriber: ContentGenerator.AudioTranscriber = self.AudioTranscriber(app_config)
        self.hf_token_general: Optional[str] = None

        self.diarization_pipeline: Optional[PyannotePipeline] = None
        # Diarization pipeline loading moved to ensure_models_loaded or a specific diarization setup method
        # to be called explicitly by main() if diarization is part of the generate_transcribed_segments flow.

    def _ensure_hf_token_general(self) -> Optional[str]: # Renamed for clarity
        """
        Ensures the general Hugging Face token is available, fetching if necessary.
        Prioritizes token from diarization config, then Kaggle Secrets.
        Returns:
            Optional[str]: The Hugging Face token, or None if unavailable.
        """
        if self.hf_token_general is not None: # Check if already fetched (could be None or a string)
            return self.hf_token_general

        token_from_diar_config = self.app_config.content_generation.diarization.hf_token
        if self.app_config.content_generation.diarization.enabled and token_from_diar_config:
            logging.info("Using HF token from diarization configuration.")
            self.hf_token_general = token_from_diar_config
            return self.hf_token_general

        try:
            logging.info("Attempting to fetch HF token from Kaggle Secrets.")
            self.hf_token_general = UserSecretsClient().get_secret("HF_TOKEN")
            if self.hf_token_general:
                 logging.info("Successfully fetched HF token from Kaggle Secrets.")
            else: # Secret exists but is empty
                 logging.warning("HF_TOKEN from Kaggle Secrets is empty.")
                 self.hf_token_general = None # Explicitly set to None
            return self.hf_token_general
        except Exception as e: # Handles UserSecretsClient().get_secret("HF_TOKEN") not found or other errors
            logging.warning(f"Could not fetch HF_TOKEN from Kaggle Secrets: {e}. Proceeding without token if models are public.")
            self.hf_token_general = None
            return None

    def ensure_models_loaded(self, load_transcription: bool = True, load_translation: bool = True, load_diarization: bool = True) -> None:
        """
        Memuat model yang diperlukan secara selektif.
        """
        token = self._ensure_hf_token_general()

        if load_transcription and not self.transcriber.pipe:
            self.transcriber.load_model(token)

        if load_translation and not self.translation_engine.model:
            self.translation_engine.load(token)

        if load_diarization and self.app_config.content_generation.diarization.enabled and not self.diarization_pipeline:
            # Diarization pipeline loading logic
            try:
                # Token for diarization specifically (could be different or same as general)
                hf_token_diar = self.app_config.content_generation.diarization.hf_token or token
                if not hf_token_diar:
                    logging.warning("HF Token not available for PyAnnote. Disabling diarization for this session.")
                    self.app_config.content_generation.diarization.enabled = False
                else:
                    logging.info(f"Loading PyAnnote Diarization pipeline: {self.app_config.content_generation.diarization.pyannote_model}")
                    self.diarization_pipeline = PyannotePipeline.from_pretrained(
                        self.app_config.content_generation.diarization.pyannote_model,
                        use_auth_token=hf_token_diar
                    )
                    device_to_use = torch.device(self.app_config.content_generation.device)
                    self.diarization_pipeline.to(device_to_use)
                    logging.info(f"PyAnnote Diarization pipeline loaded: {self.app_config.content_generation.diarization.pyannote_model} on device {device_to_use}")
            except Exception as e:
                logging.error(f"Failed to load PyAnnote Diarization pipeline: {e}. Disabling diarization for this session.", exc_info=True)
                self.app_config.content_generation.diarization.enabled = False
    
    def generate_transcribed_segments(self, video_input_path: str, transcription_checkpoint_path: str) -> Optional[List[Dict]]:
        """
        Orchestrates audio extraction, diarization (if enabled), and transcription.
        Handles checkpointing for transcribed segments.
        Returns:
            Optional[List[Dict]]: List of diarized transcribed segments, or None if process fails.
        """
        # 1. Check for existing transcribed segments checkpoint
        if self.app_config.checkpointing.enabled and os.path.exists(transcription_checkpoint_path):
            try:
                logging.info(f"Checkpoint: Loading transcribed segments from {transcription_checkpoint_path}")
                with open(transcription_checkpoint_path, 'r', encoding='utf-8') as f:
                    loaded_segments = json.load(f)
                if not loaded_segments: raise ValueError("Checkpoint file is empty.")
                logging.info(f"Loaded {len(loaded_segments)} segments from transcription checkpoint.")
                return loaded_segments
            except Exception as e:
                logging.warning(f"Could not load from {transcription_checkpoint_path}: {e}. Re-processing.")

        # 2. Ensure necessary models are loaded (ASR and Diarization if enabled)
        self.ensure_models_loaded(load_transcription=True, load_translation=False, load_diarization=True)

        # 3. Perform audio extraction, diarization, transcription
        temp_audio_path = os.path.join(self.app_config.paths.working_directory, "temp_audio_for_transcription.wav")
        diarized_transcribed_segments: List[Dict] = []

        try:
            if not os.path.exists(video_input_path):
                logging.error(f"File video input tidak ditemukan: {video_input_path}")
                raise FileNotFoundError(f"File video input tidak ditemukan: {video_input_path}")

            logging.info(f"Mengekstrak audio dari {video_input_path} ke {temp_audio_path}")
            video_clip = mp.VideoFileClip(video_input_path)
            video_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            video_clip.close()
            logging.info("Ekstraksi audio berhasil.")

            speaker_segments: List[Dict] = []
            if self.app_config.content_generation.diarization.enabled and self.diarization_pipeline:
                logging.info(f"Melakukan diarization speaker pada {temp_audio_path}...")
                diarization_result = self.diarization_pipeline(temp_audio_path)
                for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                    speaker_segments.append({"speaker": speaker_label, "start": turn.start, "end": turn.end})
                if speaker_segments: speaker_segments.sort(key=lambda x: x['start'])
                logging.info(f"Diarization menemukan {len(speaker_segments)} segmen pembicara.")

            if not self.app_config.content_generation.diarization.enabled or not speaker_segments:
                logging.info("Diarization dinonaktifkan atau tidak menghasilkan segmen. Membuat segmen dummy tunggal.")
                with mp.AudioFileClip(temp_audio_path) as audio_clip_for_duration:
                    audio_duration = audio_clip_for_duration.duration
                speaker_segments = [{"speaker": "SPEAKER_00", "start": 0, "end": audio_duration}]

            logging.info(f"Memulai transkripsi untuk keseluruhan audio: {temp_audio_path}")
            whisper_chunks = self.transcriber.transcribe(temp_audio_path)
            if not whisper_chunks:
                logging.error("Transkripsi (Whisper) tidak menghasilkan segmen/chunk.")
                raise RuntimeError("Transkripsi (Whisper) tidak menghasilkan segmen/chunk.")

            for chunk in whisper_chunks:
                chunk_mid_time = (chunk['timestamp'][0] + chunk['timestamp'][1]) / 2.0
                speaker_id = get_speaker_for_timestamp(chunk_mid_time, speaker_segments) \
                    if self.app_config.content_generation.diarization.enabled and speaker_segments \
                    else "SPEAKER_00"
                diarized_transcribed_segments.append({
                    'speaker': speaker_id, 'text': chunk['text'], 'timestamp': chunk['timestamp']
                })
            logging.info(f"Berhasil menggabungkan {len(diarized_transcribed_segments)} segmen transkripsi dengan info speaker.")

            if self.app_config.checkpointing.enabled and diarized_transcribed_segments:
                logging.info(f"Menyimpan checkpoint transkripsi ke {transcription_checkpoint_path}")
                with open(transcription_checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(diarized_transcribed_segments, f, indent=2, ensure_ascii=False)

            return diarized_transcribed_segments

        except Exception as e:
            logging.error(f"Gagal dalam proses generate_transcribed_segments: {e}", exc_info=True)
            return None # Return None on failure
        finally:
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logging.info(f"File audio sementara {temp_audio_path} berhasil dihapus.")
                except OSError as e:
                    logging.error(f"Gagal menghapus file audio sementara {temp_audio_path}: {e}", exc_info=True)

    class AudioTranscriber:
        """
        Mengelola transkripsi audio menggunakan model Whisper.
        """
        def __init__(self, app_config: AppConfigModel) -> None:
            """
            Inisialisasi AudioTranscriber dengan FasterWhisper.
            Args:
                app_config (AppConfigModel): Konfigurasi aplikasi Pydantic.
            """
            self.app_config: AppConfigModel = app_config
            self.cfg_content_gen: ContentGenerationConfigModel = app_config.content_generation
            self.model_name: str = self.cfg_content_gen.whisper_model
            self.device: str = self.cfg_content_gen.device
            # self.hf_token: Optional[str] = None # Token akan di-pass ke load_model
            self.pipe: Optional[WhisperModel] = None # Ini akan menjadi instance WhisperModel dari faster-whisper
            logging.info(f"AudioTranscriber diinisialisasi untuk model: {self.model_name} di device: {self.device}")

        def load_model(self, hf_token: Optional[str]) -> None: # Token di-pass sebagai argumen
            """
            Memuat model FasterWhisper.
            Args:
                hf_token (Optional[str]): Token Hugging Face (jika model privat).
                                          faster-whisper mungkin tidak selalu membutuhkannya untuk model publik.
            """
            logging.info(f"Memuat model FasterWhisper: '{self.model_name}' ke device: '{self.device}'...")
            try:
                # Tentukan compute_type berdasarkan device
                compute_type: str = "float16" if "cuda" in self.device.lower() else "int8"
                logging.info(f"Menggunakan compute_type: {compute_type} untuk FasterWhisper.")

                # faster-whisper menangani download model dari HF Hub jika model_name adalah ID HF.
                # Token tidak secara eksplisit ada di signature WhisperModel, tapi library mungkin menggunakan env vars
                # atau konfigurasi HF global jika model memerlukan otentikasi.
                # Untuk penggunaan token eksplisit dengan model privat, mungkin perlu pre-download model
                # atau menggunakan HuggingFaceEndpointString jika library mendukungnya.
                # Untuk sekarang, kita asumsikan model publik atau token dihandle oleh env/HF login.
                self.pipe = WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
                logging.info(f"Model FasterWhisper '{self.model_name}' berhasil dimuat di device '{self.device}'.")
            except Exception as e:
                logging.error(f"Gagal memuat model FasterWhisper '{self.model_name}': {e}", exc_info=True)
                self.pipe = None # Pastikan pipe None jika gagal
                # Pertimbangkan untuk raise error di sini jika model penting untuk operasi selanjutnya
                # raise RuntimeError(f"Gagal memuat model FasterWhisper: {e}") from e

        def transcribe(self, audio_path: str) -> List[Dict]:
            """
            Mentranskripsi audio dari file yang diberikan menggunakan FasterWhisper.
            Args:
                audio_path (str): Path ke file audio.
            Returns:
                List[Dict]: Daftar segmen transkripsi dengan 'text' dan 'timestamp'.
            """
            if not self.pipe:
                logging.error("Model FasterWhisper belum dimuat. Panggil load_model terlebih dahulu.")
                return [] # Kembalikan list kosong atau raise error

            logging.info(f"Memulai transkripsi dengan FasterWhisper untuk: {audio_path}...")
            whisper_chunks: List[Dict] = []
            try:
                # Referensi API faster-whisper: model.transcribe(audio, language=None, task=None, ...)
                # language=None berarti auto-detect. task="transcribe" adalah default.
                # beam_size adalah parameter umum untuk tuning.
                segments_generator, info = self.pipe.transcribe(
                    audio_path,
                    beam_size=5, # Default umum, bisa dibuat configurable
                    word_timestamps=False, # Kita hanya butuh timestamp segmen
                    language=None, # Auto-detect bahasa, atau set "en" jika selalu Inggris
                    task="transcribe"
                )

                logging.info(f"Bahasa terdeteksi oleh FasterWhisper: {info.language} dengan probabilitas {info.language_probability:.2f}")

                for segment in segments_generator:
                    whisper_chunks.append({
                        "text": segment.text.strip(),
                        "timestamp": [segment.start, segment.end] # segment.start dan .end adalah float dalam detik
                    })
                logging.info(f"Transkripsi dengan FasterWhisper selesai. Ditemukan {len(whisper_chunks)} segmen.")
            except Exception as e:
                logging.error(f"Error selama transkripsi dengan FasterWhisper: {e}", exc_info=True)
                return [] # Kembalikan list kosong jika ada error

            return whisper_chunks

    class TranslationEngine:
        """
        Mengelola proses terjemahan teks dan penulisan file SRT.
        """
        def __init__(self, app_config: AppConfigModel) -> None:
            """
            Inisialisasi TranslationEngine.
            Args:
                app_config (AppConfigModel): Konfigurasi aplikasi Pydantic.
            """
            self.app_config: AppConfigModel = app_config
            self.cfg_content_gen: ContentGenerationConfigModel = app_config.content_generation
            self.paths_config: PathsConfigModel = app_config.paths
            self.mapping_json_path: str = os.path.join(self.paths_config.dataset_path, self.paths_config.mapping_json_filename)

            self.mapping: Dict[str, str] = {}
            self.model: Optional[AutoModelForSeq2SeqLM] = None
            self.tokenizer: Optional[AutoTokenizer] = None

        def load(self, token: Optional[str]) -> None:
            """
            Memuat model terjemahan, tokenizer, dan peta istilah khusus.
            Args:
                token (Optional[str]): Token Hugging Face.
            """
            logging.info(f"Memuat model terjemahan: '{self.cfg_content_gen.translation_model}' dan file mapping: {self.mapping_json_path}")
            try:
                with open(self.mapping_json_path, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                self.mapping = dict(sorted(self.mapping.items(), key=lambda item: len(item[0]), reverse=True))
            except FileNotFoundError:
                logging.warning(f"File mapping JSON tidak ditemukan di {self.mapping_json_path}. Melanjutkan tanpa mapping.")
                self.mapping = {}
            except json.JSONDecodeError:
                logging.error(f"Error mendekode JSON dari file mapping: {self.mapping_json_path}. Melanjutkan tanpa mapping.")
                self.mapping = {}

            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg_content_gen.translation_model, token=token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg_content_gen.translation_model, token=token).to(self.cfg_content_gen.device)
            logging.info("Mesin Terjemahan (model dan mapping) berhasil dimuat.")

        def _translate_sentence(self, sentence_text: str, previous_translated_indonesian_text: Optional[str] = None) -> str:
            """
            Menerjemahkan satu kalimat EN ke ID dan menerapkan mapping.
            Args:
                sentence_text (str): Kalimat Inggris untuk diterjemahkan.
                previous_translated_indonesian_text (Optional[str]): Terjemahan Indonesia dari kalimat sebelumnya (untuk potensi konteks).
            Returns:
                str: Kalimat terjemahan dalam Bahasa Indonesia.
            """
            if not sentence_text.strip(): return ""
            if not self.model or not self.tokenizer:
                logging.error("Model/tokenizer terjemahan belum dimuat.")
                raise RuntimeError("Model/tokenizer terjemahan belum dimuat.")

            if previous_translated_indonesian_text:
                logging.debug(f"Menerima konteks sebelumnya (ID): '{previous_translated_indonesian_text[:50]}...'")
            else:
                logging.debug("Tidak ada konteks sebelumnya yang diterima untuk terjemahan.")

            # Komentar mengenai Kontekstualisasi:
            # Implementasi kontekstualisasi yang sebenarnya sangat bergantung pada model NMT.
            # Model standar (seperti Opus-MT) yang dipanggil via `model.generate()` biasanya tidak memiliki
            # cara langsung untuk menerima konteks target-side (terjemahan sebelumnya) melalui parameter sederhana.
            # Fitur ini mungkin memerlukan arsitektur model khusus atau fine-tuning.
            # Perubahan saat ini hanya menyiapkan alur data untuk konteks; tidak mengubah cara `model.generate()` dipanggil.
            # Jika model `xgatsby/opus-mt-en-id-avt` mendukung parameter konteks khusus, itu perlu ditambahkan di sini.
            # Untuk saat ini, diasumsikan model tidak menggunakan `previous_translated_indonesian_text` secara eksplisit.
            logging.debug("Catatan: Terjemahan kontekstual sebenarnya bergantung pada kemampuan model NMT yang digunakan.")

            inputs = self.tokenizer(sentence_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.cfg_content_gen.device)
            # Panggilan ke model.generate() tetap sama.
            translated_tokens = self.model.generate(**inputs, max_length=512)
            indonesian_translation: str = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            processed_translation: str = indonesian_translation
            if self.mapping: # Hanya lakukan jika mapping ada
                for term_to_find, replacement in self.mapping.items():
                    processed_translation = re.sub(re.escape(term_to_find), replacement, processed_translation, flags=re.IGNORECASE)
            return processed_translation

        def process_and_write_srt(self, segments: List[Dict], output_path: str) -> None: # output_path dari argumen
            """
            Memproses segmen transkripsi menjadi kalimat logis, menerjemahkannya, dan menulis hasilnya ke file SRT.
            Args:
                segments (List[Dict]): Daftar segmen dari Whisper.
                output_path (str): Path untuk menyimpan file SRT yang dihasilkan.
            """
            if not segments:
                logging.warning("Tidak ada segmen untuk diproses menjadi SRT. File SRT kosong akan ditulis.")
                with open(output_path, 'w', encoding='utf-8') as f: f.write("")
                return

            logical_sentences: List[Dict] = []
            current_sentence_chunks_info: List[Dict] = [] # Stores {'text':..., 'timestamp':..., 'speaker':...}

            for idx, seg in enumerate(segments): # segments now include 'speaker'
                current_sentence_chunks_info.append({
                    'text': seg['text'],
                    'timestamp': seg['timestamp'],
                    'speaker': seg.get('speaker', 'SPEAKER_00') # Default if speaker somehow missing
                })

                is_last_segment: bool = (idx == len(segments) - 1)
                ends_with_punctuation: bool = seg['text'].strip().endswith(('.', '?', '!'))

                if (ends_with_punctuation or is_last_segment) and current_sentence_chunks_info:
                    # Construct full sentence text from accumulated chunks
                    full_sentence_text: str = " ".join(chunk['text'] for chunk in current_sentence_chunks_info).strip()
                    full_sentence_text = re.sub(r'\s*([.?!])\s*', r'\1 ', full_sentence_text)
                    full_sentence_text = re.sub(r'\s+', ' ', full_sentence_text).strip()

                    if full_sentence_text:
                        start_time_s: float = current_sentence_chunks_info[0]['timestamp'][0]
                        end_time_s: float = current_sentence_chunks_info[-1]['timestamp'][1]
                        # Assign speaker of the first chunk to the whole logical sentence
                        sentence_speaker: str = current_sentence_chunks_info[0]['speaker']

                        logical_sentences.append({
                            'text': full_sentence_text,
                            'timestamp': [start_time_s, end_time_s],
                            'speaker': sentence_speaker, # Store the determined speaker for the sentence
                            'original_chunks_count': len(current_sentence_chunks_info)
                        })
                    current_sentence_chunks_info = [] # Reset for next sentence
            
            logging.info(f"Menulis {len(logical_sentences)} kalimat logis ke file SRT: {output_path}")
            previous_translation_for_context: Optional[str] = None # Inisialisasi konteks
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, sentence_data in enumerate(logical_sentences):
                    sentence_to_translate: str = sentence_data['text']
                    # Panggil _translate_sentence dengan konteks dari iterasi sebelumnya
                    translated_text: str = self._translate_sentence(
                        sentence_to_translate,
                        previous_translation_for_context
                    )
                    # Simpan terjemahan saat ini untuk digunakan sebagai konteks pada iterasi berikutnya
                    previous_translation_for_context = translated_text

                    final_text_for_srt: str = translated_text # Default text
                    # Terapkan prefix pembicara jika diarization diaktifkan
                    if self.app_config.content_generation.diarization.enabled:
                        speaker_id_str: str = sentence_data.get('speaker', 'SPEAKER_00')
                        prefix_format: str = self.app_config.content_generation.diarization.speaker_prefix_format

                        if prefix_format:
                            try:
                                prefix: str = prefix_format.format(speaker_id=str(speaker_id_str))
                                final_text_for_srt = f"{prefix}{translated_text}"
                            except KeyError:
                                logging.warning(f"Error format string untuk speaker_prefix_format: '{prefix_format}'. Menggunakan teks terjemahan mentah.")
                        # else: prefix_format kosong, tidak ada yang dilakukan

                    start_s, end_s = sentence_data['timestamp']
                    start_time_str = str(datetime.timedelta(seconds=start_s)).split('.')[0] + f",{int((start_s % 1) * 1000):03}"
                    end_time_str = str(datetime.timedelta(seconds=end_s)).split('.')[0] + f",{int((end_s % 1) * 1000):03}"
                    f.write(f"{i+1}\n{start_time_str} --> {end_time_str}\n{final_text_for_srt.strip()}\n\n")

# --- Bagian 3: Kelas-Kelas Pemoles Profesional (Tahap 2) ---

# Kelas @dataclass SubtitleStandards (lama) sudah dihapus secara implisit karena tidak ada lagi di sini.
# Penggunaannya digantikan oleh model Pydantic SubtitleStandardsModel dari config.py,
# yang diakses melalui app_config.subtitle_polishing.

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

    def __init__(self, standards_config: SubtitleStandardsModel) -> None: # Diperbarui untuk menerima SubtitleStandardsModel
        """
        Inisialisasi IntelligentLineBreaker.
        Args:
            standards_config (SubtitleStandardsModel): Objek Pydantic standar subtitle yang akan digunakan.
        """
        self.standards: SubtitleStandardsModel = standards_config # Menggunakan model Pydantic
        logging.info("IntelligentLineBreaker diinisialisasi dengan aturan gramatikal dan penyeimbangan yang ditingkatkan, menggunakan model Pydantic.")

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
    def __init__(self, app_config: AppConfigModel) -> None: # Menggunakan AppConfigModel
        """
        Inisialisasi ProfessionalSubtitleProcessor.
        Args:
            app_config (AppConfigModel): Konfigurasi aplikasi Pydantic.
        """
        self.app_config: AppConfigModel = app_config
        self.standards: SubtitleStandardsModel = app_config.subtitle_polishing # Mengambil dari AppConfigModel
        self.line_breaker: IntelligentLineBreaker = IntelligentLineBreaker(self.standards) # Meneruskan model standar
        logging.info("ProfessionalSubtitleProcessor diinisialisasi dengan model konfigurasi Pydantic.")

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

def execute_pipeline(app_config: AppConfigModel) -> None:
    """
    Fungsi utama yang menjalankan seluruh alur kerja pembuatan dan pemolesan subtitle,
    menggunakan objek konfigurasi Pydantic yang sudah dimuat.
    Args:
        app_config (AppConfigModel): Konfigurasi aplikasi yang sudah divalidasi.
    """
    # === Logging Setup ===
    # Pastikan direktori untuk log ada. working_directory seharusnya sudah dibuat oleh validator Pydantic
    # atau jika belum, dibuat di sini.
    os.makedirs(app_config.paths.working_directory, exist_ok=True)
    log_file_path = os.path.join(app_config.paths.working_directory, app_config.paths.log_filename)
    
    # Hapus handler logging sebelumnya untuk menghindari duplikasi jika fungsi ini dipanggil berkali-kali
    # (misalnya dalam sesi Streamlit yang panjang atau pengujian berulang).
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close() # Tutup handler sebelum menghapusnya

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(module)s - %(message)s',
        force=True, # Paksa konfigurasi ulang
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8', mode='w'), # Mode 'w' untuk log baru setiap run
            logging.StreamHandler(sys.stdout) # Juga log ke console
        ]
    )
    logging.info(f"Pipeline AVT Subtitler Pro Dimulai. Log file: {log_file_path}")

    # Untuk debugging, tampilkan sebagian konfigurasi (hati-hati dengan data sensitif seperti token)
    try:
        config_dump_dict = app_config.model_dump()
        # Sembunyikan token dari log
        if 'diarization' in config_dump_dict.get('content_generation', {}) and \
           'hf_token' in config_dump_dict['content_generation']['diarization']:
            config_dump_dict['content_generation']['diarization']['hf_token'] = "****HIDDEN****"
        logging.debug(f"Menggunakan konfigurasi: {json.dumps(config_dump_dict, indent=2, default=str)}")
    except Exception as e:
        logging.warning(f"Gagal membuat dump konfigurasi untuk logging: {e}")


    # === Path Setup dari app_config ===
    paths_cfg: PathsConfigModel = app_config.paths
    working_dir: str = paths_cfg.working_directory # Sudah divalidasi/dibuat oleh Pydantic

    transcription_checkpoint_file: str = os.path.join(working_dir, paths_cfg.transcription_checkpoint_filename)
    raw_srt_output_path: str = os.path.join(working_dir, paths_cfg.raw_srt_filename)
    final_srt_output_path: str = os.path.join(working_dir, paths_cfg.final_srt_filename)
    video_input_path: str = app_config.content_generation.video_input_path # Bisa dinamis dari UI

    # === Inisialisasi Komponen Utama ===
    # Model akan dimuat sesuai kebutuhan oleh metode ensure_models_loaded()
    content_generator = ContentGenerator(app_config=app_config)
    professional_processor = ProfessionalSubtitleProcessor(app_config=app_config)

    # === TAHAP 1: GENERASI KONTEN MENTAH ===
    logging.info("--- Memulai TAHAP 1: Generasi Konten Mentah ---")

    segments_for_translation: Optional[List[Dict]] = None
    stage1_fully_skipped: bool = False # Flag untuk menandai jika seluruh tahap 1 dilewati

    # Periksa checkpoint file SRT mentah (prioritas tertinggi)
    if app_config.checkpointing.enabled and os.path.exists(raw_srt_output_path) and os.path.getsize(raw_srt_output_path) > 0:
        logging.info(f"Checkpoint: File SRT mentah '{raw_srt_output_path}' ditemukan. Melewati Tahap 1 (Diarization, Transkripsi, Terjemahan).")
        stage1_fully_skipped = True
    else:
        # Jika SRT mentah tidak ada, coba dapatkan segmen transkripsi
        # (dari checkpoint atau proses baru melalui generate_transcribed_segments)

        # Pastikan model yang dibutuhkan untuk transkripsi & diarization dimuat
        content_generator.ensure_models_loaded(load_transcription=True, load_translation=False, load_diarization=True)

        segments_for_translation = content_generator.generate_transcribed_segments(
            video_input_path=video_input_path, # Path video aktual
            transcription_checkpoint_path=transcription_checkpoint_file
        )

        if segments_for_translation:
            logging.info("--- Memulai TAHAP 1b: Penerjemahan ---")
            # Pastikan model terjemahan dimuat sebelum digunakan
            content_generator.ensure_models_loaded(load_transcription=False, load_translation=True, load_diarization=False)

            content_generator.translation_engine.process_and_write_srt(
                segments=segments_for_translation,
                output_path=raw_srt_output_path
            )
            logging.info(f"✅ TAHAP 1b SELESAI. File '{raw_srt_output_path}' telah dibuat.")
        else: # Gagal mendapatkan segmen transkripsi
            logging.critical("❌ TAHAP 1 GAGAL: Tidak ada segmen untuk diproses setelah transkripsi/diarization.")
            return # Keluar jika tidak ada segmen

    if not stage1_fully_skipped:
        logging.info("✅ TAHAP 1 (Generasi Konten Mentah) SELESAI.")
    # Jika stage1_fully_skipped, pesan sudah dicatat sebelumnya.

    # === TAHAP 2: PEMOLESAN PROFESIONAL ===
    logging.info("\n--- Memulai TAHAP 2: Pemolesan Profesional ---")
    
    if not os.path.exists(raw_srt_output_path) or os.path.getsize(raw_srt_output_path) == 0:
        logging.critical(f"❌ TAHAP 2 GAGAL: File input SRT mentah '{raw_srt_output_path}' tidak ditemukan atau kosong setelah Tahap 1.")
        return

    try:
        final_subtitles: List[Dict] = professional_processor.process_from_file(raw_srt_output_path)
        professional_processor.write_srt_file(final_subtitles, final_srt_output_path)
        logging.info(f"✅ TAHAP 2 SELESAI. File subtitle profesional disimpan di {final_output_srt}")
    except Exception as e:
        logging.critical(f"❌ TAHAP 2 GAGAL KRITIS. Error: {e}", exc_info=True)
        # Tidak perlu return di sini, error sudah tercatat.

    if os.path.exists(final_srt_output_path):
        logging.info(f"✅ Pipeline Selesai. File subtitle akhir di: {final_srt_output_path}")
    else:
        logging.error("Pipeline selesai, namun file SRT akhir tidak ditemukan.")


if __name__ == "__main__":
    # Panggil fungsi main jika script dijalankan secara langsung.
    # Ini akan memuat konfigurasi dari config.yaml default dan menjalankan pipeline.
    loaded_config = load_app_config() # Muat konfigurasi dari file YAML default
    if loaded_config:
        execute_pipeline(app_config=loaded_config)
    else:
        # Logging sudah dilakukan di load_app_config jika gagal
        print("Gagal menjalankan pipeline karena konfigurasi tidak bisa dimuat.", file=sys.stderr)