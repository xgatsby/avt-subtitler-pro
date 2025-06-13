# =======================================================================
# AVT SUBTITLER PRO - FINAL ALL-IN-ONE SCRIPT (No spaCy)
# Version: 6.0.1 (Final Logic and Structure)
# =======================================================================

# --- Bagian 1: Impor & Konfigurasi Awal ---
import os, sys, json, re, time, logging, datetime, shutil
from typing import List, Dict, Tuple, Optional
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
    DATASET_PATH = "/kaggle/input/avt-subtitler-pro-assets"
    VIDEO_INPUT_PATH = os.path.join(DATASET_PATH, "input.mp4")
    MAPPING_JSON_PATH = os.path.join(DATASET_PATH, "mapping.json")
    RAW_SRT_OUTPUT_PATH = "/kaggle/working/raw_subtitle.srt"
    LOG_FILE_PATH = "/kaggle/working/avt_subtitler_pro.log"
    WHISPER_MODEL = "xgatsby/whisper-large-v3-avt-workshop"
    TRANSLATION_MODEL = "xgatsby/opus-mt-en-id-avt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ContentGenerator:
    def __init__(self):
        self.cfg = ContentConfig()
        self.translation_engine = self.TranslationEngine(self.cfg)
        self.transcriber = self.AudioTranscriber(self.cfg)

    def run(self):
        logging.info("=== RAW CONTENT GENERATOR STARTING ===")
        hf_token = self._get_hf_token()
        self.transcriber.load_model(hf_token)
        transcribed_segments = self.transcriber.transcribe(self.cfg.VIDEO_INPUT_PATH)
        if not transcribed_segments:
            raise RuntimeError("Transcription produced no segments.")

        self.translation_engine.load(hf_token)
        self.translation_engine.process_and_write_srt(transcribed_segments, self.cfg.RAW_SRT_OUTPUT_PATH)
        logging.info(f"Raw subtitle file created at {self.cfg.RAW_SRT_OUTPUT_PATH}")
    
    def _get_hf_token(self):
        try:
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:
            logging.warning(f"Could not get HF_TOKEN from Kaggle Secrets: {e}. Proceeding without token.")
            return None

    class AudioTranscriber:
        def __init__(self, config):
            self.cfg = config; self.pipe = None
        def load_model(self, token: str):
            logging.info(f"Loading Whisper model '{self.cfg.WHISPER_MODEL}'...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.cfg.WHISPER_MODEL, torch_dtype=torch.float16, use_safetensors=True, token=token)
            model.to(self.cfg.DEVICE)
            processor = AutoProcessor.from_pretrained(self.cfg.WHISPER_MODEL, token=token)
            self.pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, chunk_length_s=30, batch_size=16, return_timestamps=True, torch_dtype=torch.float16, device=self.cfg.DEVICE)
        def transcribe(self, video_path: str) -> List[Dict]:
            logging.info("Transcribing audio...")
            result = self.pipe(video_path, generate_kwargs={"language": "english"})
            return result.get("chunks", [])

    class TranslationEngine:
        def __init__(self, config):
            self.cfg = config; self.mapping = {}; self.model = None; self.tokenizer = None
        
        def load(self, token: str):
            with open(self.cfg.MAPPING_JSON_PATH, 'r', encoding='utf-8') as f: self.mapping = json.load(f)
            self.mapping = dict(sorted(self.mapping.items(), key=lambda item: len(item[0]), reverse=True))
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TRANSLATION_MODEL, token=token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.TRANSLATION_MODEL, token=token).to(self.cfg.DEVICE)
            logging.info("Translation Engine ready with new robust logic.")

        def _translate_single_with_mapping(self, text: str) -> str:
            if not text.strip(): return ""
            mapped_phrases = [re.escape(k) for k in self.mapping.keys()]
            if not mapped_phrases: # Jika mapping kosong, langsung terjemahkan
                return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.cfg.DEVICE)), skip_special_tokens=True)[0]

            pattern = re.compile(r'(' + '|'.join(mapped_phrases) + r')', re.IGNORECASE)
            parts = pattern.split(text)
            
            texts_to_translate = [part for part in parts if part and part.lower() not in (k.lower() for k in self.mapping.keys())]
            
            translated_parts = []
            if texts_to_translate:
                inputs = self.tokenizer(texts_to_translate, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.cfg.DEVICE)
                translated_tokens = self.model.generate(**inputs, max_length=512)
                translated_parts = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            
            final_sentence = []
            translation_idx = 0
            for part in parts:
                if not part: continue
                match_found = False
                for k, v in self.mapping.items():
                    if k.lower() == part.lower():
                        final_sentence.append(v); match_found = True; break
                if not match_found:
                    if translation_idx < len(translated_parts):
                        final_sentence.append(translated_parts[translation_idx]); translation_idx += 1
            return "".join(final_sentence)

        def process_and_write_srt(self, segments, output_path):
            if not segments:
                logging.warning("No segments to process. Writing an empty SRT file.")
                with open(output_path, 'w') as f: f.write("")
                return
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segments):
                    translated_text = self._translate_single_with_mapping(seg['text'].strip())
                    start_s, end_s = seg['timestamp']
                    start_time = str(datetime.timedelta(seconds=start_s)).split('.')[0] + f",{int((start_s % 1) * 1000):03}"
                    end_time = str(datetime.timedelta(seconds=end_s)).split('.')[0] + f",{int((end_s % 1) * 1000):03}"
                    f.write(f"{i+1}\n{start_time} --> {end_time}\n{translated_text}\n\n")

# --- Bagian 3: Kelas-Kelas Pemoles Profesional (Tahap 2) ---

@dataclass
class SubtitleStandards:
    MAX_LINES: int = 2; MAX_CHARS_PER_LINE: int = 42; MIN_READING_SPEED: float = 15.0; MAX_READING_SPEED: float = 25.0; MIN_DURATION: float = 0.8; MAX_DURATION: float = 7.0; MIN_GAP: float = 0.083; PREFERRED_READING_SPEED: float = 21.0; merge_gap_threshold: float = 0.75
    @classmethod
    def from_config(cls, config_dict: Dict) -> 'SubtitleStandards':
        standards_config = config_dict.get('standards', {}); instance = cls()
        for key, value in standards_config.items():
            if hasattr(instance, key): setattr(instance, key, value)
        return instance

class IntelligentLineBreaker:
    def __init__(self, standards: SubtitleStandards): self.standards = standards; logging.info("LineBreaker initialized (rule-based).")
    def break_lines(self, text: str) -> str:
        max_chars = self.standards.MAX_CHARS_PER_LINE
        if not text.strip() or len(text) <= max_chars: return text
        if len(text) > max_chars * 2: text = self._truncate_text(text, max_chars * 2)
        return self._simple_breaking(text, max_chars)
    def _simple_breaking(self, text: str, max_chars: int) -> str:
        words = text.split(); line1_words = []; current_len = -1
        for i, word in enumerate(words):
            if current_len + len(word) + 1 > max_chars:
                line2_words = words[i:]; break
            line1_words.append(word); current_len += len(word) + 1
        else: line2_words = []
        line1 = ' '.join(line1_words); line2 = ' '.join(line2_words)
        return self._balance_lines(line1, line2) if line2 else line1
    def _balance_lines(self, line1: str, line2: str) -> str:
        if len(line2.split()) == 1 and len(line2) <= 5:
            line1_words = line1.split()
            if len(line1_words) > 1:
                new_line1 = ' '.join(line1_words[:-1]); new_line2 = line1_words[-1] + ' ' + line2
                if len(new_line1) <= self.standards.MAX_CHARS_PER_LINE and len(new_line2) <= self.standards.MAX_CHARS_PER_LINE: return f"{new_line1}\n{new_line2}"
        return f"{line1}\n{line2}"
    def _truncate_text(self, text: str, max_length: int) -> str:
        if len(text) <= max_length: return text
        pos = text.rfind(' ', 0, max_length - 3); return text[:pos] + "..." if pos != -1 else text[:max_length-3] + "..."

class ProfessionalSubtitleProcessor:
    def __init__(self, config: Dict):
        self.config = config or {}; self.standards = SubtitleStandards.from_config(self.config)
        self.line_breaker = IntelligentLineBreaker(self.standards)
    def process_from_file(self, input_path: str) -> List[Dict]:
        raw_subs = self._parse_srt(input_path)
        merged_subs = self._merge_short_subtitles(raw_subs)
        processed_subs = []
        for sub in merged_subs:
            formatted_text = self.line_breaker.break_lines(sub['text'])
            processed_subs.append({'start': sub['start'], 'end': sub['end'], 'text': formatted_text})
        return processed_subs
    def _parse_srt(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0: return []
        subs = pysrt.open(file_path, encoding='utf-8')
        return [{'start': (s.start.hours * 3600 + s.start.minutes * 60 + s.start.seconds + s.start.milliseconds / 1000.0),
                 'end': (s.end.hours * 3600 + s.end.minutes * 60 + s.end.seconds + s.end.milliseconds / 1000.0),
                 'text': s.text} for s in subs]
    def _merge_short_subtitles(self, subtitles: List[Dict]) -> List[Dict]:
        if not subtitles: return []
        merged = []; i = 0
        while i < len(subtitles):
            current_sub = subtitles[i].copy(); j = i + 1
            while j < len(subtitles):
                next_sub = subtitles[j]; gap = next_sub['start'] - current_sub['end']
                combined_text = current_sub['text'] + " " + next_sub['text']
                combined_duration = next_sub['end'] - current_sub['start']
                if (gap < self.standards.merge_gap_threshold and len(combined_text) < (self.standards.MAX_CHARS_PER_LINE * 2 - 10) and combined_duration < self.standards.MAX_DURATION):
                    current_sub['text'] = combined_text; current_sub['end'] = next_sub['end']; j += 1
                else: break
            merged.append(current_sub); i = j
        return merged
    def write_srt_file(self, subtitles: List[Dict], file_path: str):
        subs = pysrt.SubRipFile()
        for i, sub_data in enumerate(subtitles, 1):
            item = pysrt.SubRipItem(index=i, text=sub_data['text'])
            start_seconds = sub_data['start']; end_seconds = sub_data['end']
            s_h, s_rem = divmod(start_seconds, 3600); s_m, s_s = divmod(s_rem, 60)
            item.start.hours = int(s_h); item.start.minutes = int(s_m); item.start.seconds = int(s_s); item.start.milliseconds = int((start_seconds % 1) * 1000)
            e_h, e_rem = divmod(end_seconds, 3600); e_m, e_s = divmod(e_rem, 60)
            item.end.hours = int(e_h); item.end.minutes = int(e_m); item.end.seconds = int(e_s); item.end.milliseconds = int((end_seconds % 1) * 1000)
            subs.append(item)
        subs.save(file_path, encoding='utf-8')

# --- Bagian 4: Eksekusi Utama ---

def main():
    """Fungsi utama yang menjalankan seluruh alur kerja."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', force=True)
    
    # --- TAHAP 1: GENERASI KONTEN MENTAH ---
    logging.info("--- Memulai TAHAP 1: Generasi Konten Mentah ---")
    try:
        content_generator = ContentGenerator()
        content_generator.run()
        logging.info("✅ TAHAP 1 SELESAI. File 'raw_subtitle.srt' telah dibuat.")
    except Exception as e:
        logging.critical(f"❌ TAHAP 1 GAGAL. Error: {e}", exc_info=True)
        return

    # --- TAHAP 2: PEMOLESAN PROFESIONAL ---
    logging.info("\n--- Memulai TAHAP 2: Pemolesan Profesional ---")
    raw_input_srt = "/kaggle/working/raw_subtitle.srt"
    final_output_srt = "/kaggle/working/FINAL_professional_subtitle.srt"
    config_file_path = "/kaggle/input/avt-subtitler-pro-assets/config.yaml"
    
    if not os.path.exists(raw_input_srt):
        logging.critical(f"❌ TAHAP 2 GAGAL: File input '{raw_input_srt}' tidak ditemukan.")
        return

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
        processor = ProfessionalSubtitleProcessor(config)
        final_subtitles = processor.process_from_file(raw_input_srt)
        processor.write_srt_file(final_subtitles, final_output_srt)
        logging.info(f"✅ TAHAP 2 SELESAI. File profesional disimpan di {final_output_srt}")
    except Exception as e:
        logging.critical(f"❌ TAHAP 2 GAGAL. Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()