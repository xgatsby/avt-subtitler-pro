from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional # Keep for potential future use, though not strictly needed for these models

# Default paths and model names, mirroring the old ContentConfig structure
DEFAULT_DATASET_PATH = "/kaggle/input/avt-subtitler-pro-assets"
DEFAULT_VIDEO_INPUT_NAME = "input.mp4" # Assuming this was the intended use

class SubtitleStandardsModel(BaseModel):
    """
    Pydantic model for subtitle processing standards.
    Mirrors the fields from the old SubtitleStandards dataclass.
    """
    MAX_LINES: int = 2
    MAX_CHARS_PER_LINE: int = 42
    MIN_READING_SPEED: float = 15.0  # Characters per second (CPS)
    MAX_READING_SPEED: float = 25.0  # CPS
    MIN_DURATION: float = 0.8        # Seconds
    MAX_DURATION: float = 7.0        # Seconds
    MIN_GAP: float = 0.083           # Seconds (approx 2 frames at 24fps)
    PREFERRED_READING_SPEED: float = 21.0 # CPS, used for splitting calculations
    merge_gap_threshold: float = 0.75    # Seconds, for merging subtitles


class ContentConfigModel(BaseModel):
    """
    Pydantic model for content generation configurations.
    Mirrors fields from the old ContentConfig class and adds new ones.
    """
    DATASET_PATH: str = DEFAULT_DATASET_PATH
    VIDEO_INPUT_PATH: str = Field(default_factory=lambda: os.path.join(DEFAULT_DATASET_PATH, DEFAULT_VIDEO_INPUT_NAME)) # Needs os import if used here directly
    MAPPING_JSON_PATH: str = Field(default_factory=lambda: os.path.join(DEFAULT_DATASET_PATH, "mapping.json"))
    RAW_SRT_OUTPUT_PATH: str = "/kaggle/working/raw_subtitle.srt"
    LOG_FILE_PATH: str = "/kaggle/working/avt_subtitler_pro.log"
    WHISPER_MODEL: str = "xgatsby/whisper-large-v3-avt-workshop"
    TRANSLATION_MODEL: str = "xgatsby/opus-mt-en-id-avt"
    DEVICE: str = "cuda" # Placeholder, will be updated by torch.cuda.is_available() in main logic if needed
    SPEAKER_LABEL_FORMAT: str = "- {speaker}: " # Format for speaker labels in diarization

    # Need to import os for Field default_factory if they depend on os.path.join
    # This is a bit tricky with Pydantic models if os is not available at import time.
    # A common pattern is to resolve these paths dynamically after loading the model.
    # For now, I'll define them as they were, assuming os is available, or adjust later in main.py.
    # A better way for paths that depend on other fields or external modules (like os)
    # is to use root_validator or initialize them in the application logic after loading.
    # Given the current structure, they will be string literals as defined, or factory must be self-contained.
    # To avoid direct `os` dependency here, will make them simple strings for now or use post-init.
    # For simplicity now, VIDEO_INPUT_PATH and MAPPING_JSON_PATH will be set as simple string defaults
    # as they were in the original ContentConfig. The dynamic os.path.join was not in the original ContentConfig defaults.

    # Re-simplifying path defaults to avoid 'os' import here, assuming they are fixed or constructed later.
    VIDEO_INPUT_PATH_PLACEHOLDER: str = "input.mp4" # Will be joined with DATASET_PATH later
    MAPPING_JSON_FILENAME: str = "mapping.json"    # Will be joined with DATASET_PATH later


# Corrected ContentConfigModel with simplified paths that don't require os at definition time
import os # Now it's fine to import os as we'll use it in a controlled way or accept it as a module dependency

class ContentConfigModel(BaseModel):
    DATASET_PATH: str = "/kaggle/input/avt-subtitler-pro-assets"
    VIDEO_INPUT_PATH: str = os.path.join(DATASET_PATH, "input.mp4") # This will work if this file is imported where os is available
    MAPPING_JSON_PATH: str = os.path.join(DATASET_PATH, "mapping.json")
    RAW_SRT_OUTPUT_PATH: str = "/kaggle/working/raw_subtitle.srt"
    LOG_FILE_PATH: str = "/kaggle/working/avt_subtitler_pro.log"
    WHISPER_MODEL: str = "xgatsby/whisper-large-v3-avt-workshop"
    TRANSLATION_MODEL: str = "xgatsby/opus-mt-en-id-avt"
    DEVICE: str = "cuda" # Placeholder, should be determined dynamically in main
    SPEAKER_LABEL_FORMAT: str = "- {speaker}: "


class AppConfigModel(BaseModel):
    """
    Top-level Pydantic model for the application, nesting other configurations.
    """
    content: ContentConfigModel = Field(default_factory=ContentConfigModel)
    standards: SubtitleStandardsModel = Field(default_factory=SubtitleStandardsModel)

    class Config:
        validate_assignment = True # Useful for mutable nested models if changed post-init
        # For Pydantic V2, use model_config = {'validate_assignment': True}
        pass
