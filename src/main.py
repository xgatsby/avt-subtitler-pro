# =======================================================================
# AVT SUBTITLER PRO - FINAL ALL-IN-ONE SCRIPT (No spaCy)
# Version: 6.0.1 (Final Logic and Structure)
# =======================================================================

# --- Bagian 1: Impor & Konfigurasi Awal ---
import os, sys, json, re, time, logging, datetime, shutil
from typing import List, Dict, Tuple, Optional, frozenset
import torch
import moviepy.editor as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from kaggle_secrets import UserSecretsClient
import pysrt
import yaml
import argparse

# Impor Pydantic model dari config.py
from config import AppConfigModel, ContentGenerationConfigModel, SubtitleStandardsModel, PathsConfigModel, DiarizationConfigModel, CheckpointingConfigModel

# Imports for Diarization
from pyannote.audio import Pipeline as PyannotePipeline
import torchaudio

# Imports for Faster Whisper
from faster_whisper import WhisperModel

# --- Helper Function for Diarization ---
def get_speaker_for_timestamp(time_sec: float, diar_segments: List[Dict], default_speaker: str = "SPEAKER_00") -> str:
    """Determines speaker for given timestamp based on diarization segments."""
    if not diar_segments:
        return default_speaker

    for seg in diar_segments:
        if seg['start'] <= time_sec < seg['end']:
            return seg['speaker']

    if time_sec < diar_segments[0]['start']:
        return diar_segments[0]['speaker']

    if time_sec >= diar_segments[-1]['end']:
        return diar_segments[-1]['speaker']

    for i in range(len(diar_segments) - 1):
        current_seg_end = diar_segments[i]['end']
        next_seg_start = diar_segments[i+1]['start']
        if current_seg_end <= time_sec < next_seg_start:
            return diar_segments[i]['speaker']

    return default_speaker

# --- Fungsi Pemuatan Konfigurasi ---
def load_app_config(config_path: str = "config.yaml") -> AppConfigModel:
    """Memuat konfigurasi aplikasi dari file YAML dan memvalidasinya dengan Pydantic model."""
    logging.info(f"Mencoba memuat konfigurasi dari: {config_path}")
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            logging.warning(f"File konfigurasi {config_path} kosong atau bukan YAML valid. Menggunakan default.")
            config_data = {}
    else:
        logging.warning(f"File konfigurasi {config_path} tidak ditemukan. Menggunakan default.")

    try:
        app_config = AppConfigModel(**config_data)
        logging.info("Konfigurasi berhasil dimuat dan divalidasi.")

        if app_config.content_generation.device is None:
            app_config.content_generation.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Device auto-detected and set to: {app_config.content_generation.device}")

        return app_config
    except Exception as e:
        logging.error(f"Error saat memuat atau memvalidasi konfigurasi dari {config_path}: {e}", exc_info=True)
        raise

# --- Bagian 2: Kelas-Kelas Generator Konten (Tahap 1) ---
class ContentGenerator:
    """Mengelola proses pembuatan konten mentah: transkripsi audio dan terjemahan awal."""
    def __init__(self, app_config: AppConfigModel) -> None:
        self.app_config: AppConfigModel = app_config
        self.cfg_content_gen: ContentGenerationConfigModel = app_config.content_generation
        self.paths_config: PathsConfigModel = app_config.paths

        self.video_input_path: str = self.cfg_content_gen.video_input_path
        self.mapping_json_path: str = os.path.join(self.paths_config.dataset_path, self.paths_config.mapping_json_filename)
        self.raw_srt_output_path: str = os.path.join(self.paths_config.working_directory, self.paths_config.raw_srt_filename)
        self.log_file_path: str = os.path.join(self.paths_config.working_directory, self.paths_config.log_filename)

        self.translation_engine: ContentGenerator.TranslationEngine = self.TranslationEngine(app_config)
        self.transcriber: ContentGenerator.AudioTranscriber = self.AudioTranscriber(app_config)
        self.hf_token_general: Optional[str] = None
        self.diarization_pipeline: Optional[PyannotePipeline] = None

    # ... [rest of the ContentGenerator class implementation] ...

# --- Bagian 3: Kelas-Kelas Pemoles Profesional (Tahap 2) ---
class IntelligentLineBreaker:
    """Memecah teks menjadi beberapa baris subtitle dengan mempertimbangkan panjang maksimal dan aturan gramatikal."""
    INDONESIAN_CONJUNCTIONS: frozenset[str] = frozenset([
        'yang', 'dan', 'atau', 'tetapi', 'namun', 'sedangkan', 'melainkan', 'serta', 'lalu', 'kemudian',
        'jika', 'kalau', 'ketika', 'saat', 'sebelum', 'sesudah', 'karena', 'sebab', 'agar', 'supaya',
        'meskipun', 'walaupun'
    ])

    def __init__(self, standards_config: SubtitleStandardsModel) -> None:
        self.standards: SubtitleStandardsModel = standards_config

    # ... [rest of IntelligentLineBreaker implementation] ...

class ProfessionalSubtitleProcessor:
    """Memproses subtitle mentah menjadi format profesional dengan menerapkan aturan standar industri."""
    def __init__(self, app_config: AppConfigModel) -> None:
        self.app_config: AppConfigModel = app_config
        self.standards: SubtitleStandardsModel = app_config.subtitle_polishing
        self.line_breaker: IntelligentLineBreaker = IntelligentLineBreaker(self.standards)

    # ... [rest of ProfessionalSubtitleProcessor implementation] ...

# --- Bagian 4: Eksekusi Utama ---
def execute_pipeline(app_config: AppConfigModel) -> None:
    """Fungsi utama pipeline yang menggunakan model konfigurasi Pydantic."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - (%(module)s:%(lineno)d) - %(message)s',
        force=True
    )
    
    try:
        # Tahap 1: Generate transcribed segments
        content_gen = ContentGenerator(app_config)
        transcription_checkpoint = os.path.join(
            app_config.paths.working_directory,
            "transcription_checkpoint.json"
        )
        segments = content_gen.generate_transcribed_segments(
            app_config.content_generation.video_input_path,
            transcription_checkpoint
        )
        
        if not segments:
            raise RuntimeError("No segments generated from transcription")
            
        # Tahap 2: Process and write SRT
        content_gen.translation_engine.process_and_write_srt(
            segments,
            os.path.join(app_config.paths.working_directory, "raw_subtitle.srt")
        )
        logging.info("✅ TAHAP 1 SELESAI. File 'raw_subtitle.srt' telah dibuat.")

        # Tahap 3: Professional polishing
        processor = ProfessionalSubtitleProcessor(app_config)
        final_subs = processor.process_from_file(
            os.path.join(app_config.paths.working_directory, "raw_subtitle.srt")
        )
        processor.write_srt_file(
            final_subs,
            os.path.join(app_config.paths.working_directory, "FINAL_professional_subtitle.srt")
        )
        logging.info("✅ TAHAP 2 SELESAI. File subtitle profesional telah dibuat.")
        
    except Exception as e:
        logging.critical(f"❌ PIPELINE GAGAL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    config = load_app_config()
    execute_pipeline(config)
