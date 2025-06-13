from pydantic import BaseModel, Field, FilePath, DirectoryPath, validator
from typing import List, Optional, Union, Dict
import os

class PathsConfigModel(BaseModel):
    """
    Konfigurasi untuk path dasar yang digunakan oleh aplikasi.
    Beberapa path absolut akan dibangun dari kombinasi path dasar dan nama file.
    """
    dataset_path: DirectoryPath = DirectoryPath("/kaggle/input/avt-subtitler-pro-assets") # Path ke dataset input (harus ada)
    working_directory: str = "/kaggle/working"  # Path ke direktori kerja untuk output (akan dibuat jika belum ada)

    mapping_json_filename: str = Field("mapping.json", description="Nama file untuk peta istilah EN->ID.")
    raw_srt_filename: str = Field("raw_subtitle.srt", description="Nama file untuk output SRT mentah.")
    final_srt_filename: str = Field("FINAL_professional_subtitle.srt", description="Nama file untuk output SRT final yang dipoles.")
    log_filename: str = Field("avt_subtitler_pro.log", description="Nama file untuk log aplikasi.")
    transcription_checkpoint_filename: str = Field("_transcription_checkpoint.json", description="Nama file untuk checkpoint transkripsi.")

    @validator('working_directory')
    def create_working_directory(cls, v: str) -> str:
        """Memastikan direktori kerja ada, membuatnya jika perlu."""
        os.makedirs(v, exist_ok=True)
        return v

class DiarizationConfigModel(BaseModel):
    """
    Konfigurasi untuk proses diarization (pemisahan pembicara).
    """
    enabled: bool = Field(True, description="Aktifkan atau nonaktifkan diarization.")
    hf_token: Optional[str] = Field(None, description="Token Hugging Face untuk model PyAnnote (opsional, untuk model privat atau menghindari rate limit).")
    speaker_prefix_format: str = Field("- {speaker_id}: ", description="Format prefix untuk setiap segmen pembicara (misalnya, '- SPEAKER_00: ').")
    pyannote_model: str = Field("pyannote/speaker-diarization-3.1", description="Nama model PyAnnote yang akan digunakan.")

class ContentGenerationConfigModel(BaseModel):
    """
    Konfigurasi untuk tahap generasi konten (transkripsi dan terjemahan awal).
    """
    video_input_path: str = Field("/kaggle/input/avt-subtitler-pro-assets/input.mp4", description="Path default ke video input. Akan dioverride oleh UI Streamlit.")
    whisper_model: str = Field("xgatsby/whisper-large-v3-avt-workshop", description="Model Whisper untuk transkripsi ASR.")
    translation_model: str = Field("xgatsby/opus-mt-en-id-avt", description="Model NMT untuk terjemahan EN-ID.")
    device: Optional[str] = Field(None, description="Perangkat untuk inferensi model ('cuda' atau 'cpu'). Akan dideteksi otomatis jika None.")
    diarization: DiarizationConfigModel = DiarizationConfigModel()

class SubtitleStandardsModel(BaseModel):
    """
    Konfigurasi untuk standar pemolesan subtitle, mencerminkan kelas SubtitleStandards.
    """
    max_lines: int = Field(2, description="Jumlah baris maksimum per subtitle.")
    max_chars_per_line: int = Field(42, description="Jumlah karakter maksimum per baris.")
    min_reading_speed: float = Field(15.0, description="Kecepatan membaca minimum (karakter per detik, tanpa spasi).")
    max_reading_speed: float = Field(25.0, description="Kecepatan membaca maksimum (karakter per detik, tanpa spasi).")
    min_duration: float = Field(0.8, description="Durasi minimum subtitle dalam detik.")
    max_duration: float = Field(7.0, description="Durasi maksimum subtitle dalam detik.")
    min_gap: float = Field(0.083, description="Jarak minimum antar subtitle dalam detik.")
    preferred_reading_speed: float = Field(21.0, description="Kecepatan membaca yang diutamakan untuk perhitungan durasi saat pemisahan.")
    merge_gap_threshold: float = Field(0.75, description="Batas jarak (detik) untuk menggabungkan subtitle yang berdekatan.")

class CheckpointingConfigModel(BaseModel):
    """
    Konfigurasi untuk fitur checkpointing.
    """
    enabled: bool = Field(True, description="Aktifkan atau nonaktifkan checkpointing untuk melanjutkan proses yang terputus.")
    # File checkpoint transkripsi akan disimpan di working_directory + transcription_checkpoint_filename dari PathsConfigModel

class AppConfigModel(BaseModel):
    """
    Model konfigurasi utama untuk seluruh aplikasi AVT Subtitler Pro.
    Menggabungkan semua sub-konfigurasi.
    """
    paths: PathsConfigModel = PathsConfigModel()
    content_generation: ContentGenerationConfigModel = ContentGenerationConfigModel()
    subtitle_polishing: SubtitleStandardsModel = SubtitleStandardsModel() # Menggunakan nama bagian yang lebih jelas
    checkpointing: CheckpointingConfigModel = CheckpointingConfigModel()

# Fungsi helper untuk memuat konfigurasi dari file YAML (akan diimplementasikan nanti jika diperlukan di sini)
# Untuk saat ini, diasumsikan file YAML akan dibaca dan dictionary-nya akan digunakan untuk membuat AppConfigModel
# contoh: AppConfigModel(**yaml_data)

if __name__ == "__main__":
    # Contoh cara membuat instance AppConfigModel (misalnya, dari data default atau data yang dimuat)
    # Ini hanya untuk pengujian atau demonstrasi.
    try:
        app_config = AppConfigModel()
        print("AppConfigModel created successfully:")
        print("Paths:", app_config.paths.model_dump_json(indent=2))
        print("Content Generation:", app_config.content_generation.model_dump_json(indent=2))
        print("Subtitle Polishing:", app_config.subtitle_polishing.model_dump_json(indent=2))
        print("Checkpointing:", app_config.checkpointing.model_dump_json(indent=2))

        # Contoh akses path yang dibangun (seperti yang akan dilakukan di main.py)
        print("\nContoh path yang dibangun:")
        mapping_path = os.path.join(app_config.paths.dataset_path, app_config.paths.mapping_json_filename)
        print(f"Mapping JSON Path: {mapping_path}")
        raw_srt_path = os.path.join(app_config.paths.working_directory, app_config.paths.raw_srt_filename)
        print(f"Raw SRT Output Path: {raw_srt_path}")

    except Exception as e:
        print(f"Error creating or using AppConfigModel: {e}")
        # Ini akan menunjukkan error validasi Pydantic jika ada masalah dengan path default
        # atau struktur model, yang berguna untuk debugging.
        # Khususnya, DirectoryPath akan gagal jika path default tidak ada saat script dijalankan.
        # Dalam penggunaan nyata, path seperti dataset_path harus valid.
        # working_directory dibuat oleh validator, jadi seharusnya aman.
        print("\nNote: DirectoryPath validation for 'dataset_path' might fail if the default path does not exist in the execution environment.")
