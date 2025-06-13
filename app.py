import streamlit as st
import os
import yaml
import subprocess # For running the main script (alternative if direct call fails)
import tempfile # For handling uploaded file
from config import AppConfigModel # To load and show current defaults
from pathlib import Path
import sys
import logging # For app.py's own logging
import re # For log parsing (if implemented)
import traceback

# Configure logging for the Streamlit app itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (%(module)s) - %(message)s')

# --- Add src to sys.path to allow importing from main ---
# This allows 'from main import execute_pipeline'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

execute_pipeline = None # Initialize
try:
    from main import execute_pipeline as main_execute_pipeline
    execute_pipeline = main_execute_pipeline # Assign if import is successful
    logging.info("Successfully imported `execute_pipeline` from `src.main`.")
except ImportError as e:
    logging.error(f"Failed to import `execute_pipeline` from `src.main`: {e}. Check structure of src/main.py.")
    # Keep execute_pipeline as None. UI will show an error.
except Exception as e:
    logging.error(f"An unexpected error occurred during import of `execute_pipeline`: {e}", exc_info=True)
    # Keep execute_pipeline as None.

# --- Main UI Structure ---
st.set_page_config(layout="wide")
st.title("AVT Subtitler Pro 🚀")
st.markdown("""
Welcome to AVT Subtitler Pro! This tool helps you generate and polish subtitles for your videos.
1.  **Upload your MP4 video.**
2.  **Adjust settings (optional).** Defaults are loaded from `config.yaml`.
3.  **Click "Generate Subtitles"** and wait for the process to complete.
4.  **Download** your professionally polished SRT subtitle file.
""")

# --- Load Base Configuration for UI Defaults ---
BASE_CONFIG_PATH = "config.yaml"

@st.cache_data # Cache the base config to avoid reloading on every interaction
def load_base_config_for_ui():
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            if config_dict is None: # Handle empty YAML file
                logging.warning(f"Base config file {BASE_CONFIG_PATH} is empty. Using Pydantic defaults.")
                return {}
            # Ensure nested structures exist to prevent KeyErrors later
            config_dict.setdefault('content_generation', {}).setdefault('diarization', {})
            config_dict.setdefault('subtitle_polishing', {})
            config_dict.setdefault('checkpointing', {})
            config_dict.setdefault('paths', {})
            return config_dict
    except FileNotFoundError:
        st.error(f"Base configuration file (`{BASE_CONFIG_PATH}`) not found. Using default Pydantic model values.")
        return {} # Return empty, Pydantic models in AppConfigModel will use their defaults
    except Exception as e:
        st.error(f"Error loading base configuration (`{BASE_CONFIG_PATH}`): {e}. Using default Pydantic model values.")
        return {}

ui_config_defaults = load_base_config_for_ui()

# --- UI Components ---

# File Uploader
st.header("1. Upload Video")
uploaded_file = st.file_uploader("Upload your input.mp4 video", type=["mp4"])

# Configuration Settings
st.header("2. Configure Settings (Optional)")

with st.expander("General & Advanced Settings", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Settings")
        # Get diarization defaults safely
        diar_defaults = ui_config_defaults.get('content_generation', {}).get('diarization', {})
        diarization_enabled = st.checkbox("Enable Speaker Diarization",
                                        value=diar_defaults.get('enabled', True))
        hf_token_default = diar_defaults.get('hf_token', '')
        hf_token = st.text_input("Hugging Face Token (for PyAnnote)",
                                 type="password",
                                 value=str(hf_token_default) if hf_token_default is not None else "",
                                 help="Needed if PyAnnote model is private or to avoid rate limits. Can be left blank for public models.")

        # Checkpointing default
        checkpointing_defaults = ui_config_defaults.get('checkpointing', {})
        checkpointing_enabled = st.checkbox("Enable Checkpointing",
                                            value=checkpointing_defaults.get('enabled', True),
                                            help="Resume progress from last completed step if available.")

    with col2:
        st.subheader("Subtitle Polishing Standards")
        polishing_defaults = ui_config_defaults.get('subtitle_polishing', {})
        max_lines = st.slider("Max Lines per Subtitle", 1, 3, polishing_defaults.get('max_lines', 2))
        max_chars_per_line = st.slider("Max Characters Per Line", 20, 80, polishing_defaults.get('max_chars_per_line', 42))
        min_duration = st.number_input("Min Duration (s)", 0.1, 5.0, polishing_defaults.get('min_duration', 0.8), 0.1, format="%.1f")
        max_duration = st.number_input("Max Duration (s)", 1.0, 20.0, polishing_defaults.get('max_duration', 7.0), 0.5, format="%.1f")
        min_gap = st.number_input("Min Gap Between Subs (s)", 0.0, 1.0, polishing_defaults.get('min_gap', 0.083), 0.001, format="%.3f")
        min_reading_speed = st.slider("Min Reading Speed (CPS)", 5.0, 25.0, polishing_defaults.get('min_reading_speed', 15.0), 0.5, format="%.1f")
        max_reading_speed = st.slider("Max Reading Speed (CPS)", 10.0, 40.0, polishing_defaults.get('max_reading_speed', 25.0), 0.5, format="%.1f")
        preferred_reading_speed = st.slider("Preferred Reading Speed (CPS for splitting)", 10.0, 35.0, polishing_defaults.get('preferred_reading_speed', 21.0), 0.5, format="%.1f")
        merge_gap_threshold = st.number_input("Merge Gap Threshold (s)", 0.1, 2.0, polishing_defaults.get('merge_gap_threshold', 0.75), 0.05, format="%.2f")


# "Generate Subtitles" Button
st.header("3. Generate Subtitles")
if st.button("✨ Generate Subtitles", type="primary"):
    if execute_pipeline is None:
        st.error("🚨 Critical Error: The main processing pipeline (`execute_pipeline`) could not be imported. The application cannot function. Please check the server logs or `src/main.py` structure.")
    elif uploaded_file is None:
        st.error("🚨 Please upload a video file first!")
    else:
        if diarization_enabled and not hf_token:
            st.warning("⚠️ Diarization is enabled, but no Hugging Face token was provided. This might lead to issues if the PyAnnote model requires authentication. Processing will continue...")

        with st.spinner("🚀 Processing your video... This might take a while! Please wait."):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded video to a temporary path
                temp_video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.info(f"Video saved to temporary path: {temp_video_path}")

                # Create a run-specific configuration dictionary
                # Start with a deep copy of the base defaults to avoid modifying them
                run_config_dict = yaml.safe_load(yaml.dump(ui_config_defaults)) # Simple deep copy

                # Update config with UI values
                run_config_dict['content_generation']['video_input_path'] = temp_video_path

                # Ensure nested dicts exist before trying to set values
                run_config_dict['content_generation'].setdefault('diarization', {})['hf_token'] = hf_token if hf_token else None
                run_config_dict['content_generation']['diarization']['enabled'] = diarization_enabled

                run_config_dict.setdefault('subtitle_polishing', {})
                run_config_dict['subtitle_polishing']['max_lines'] = max_lines
                run_config_dict['subtitle_polishing']['max_chars_per_line'] = max_chars_per_line
                run_config_dict['subtitle_polishing']['min_duration'] = min_duration
                run_config_dict['subtitle_polishing']['max_duration'] = max_duration
                run_config_dict['subtitle_polishing']['min_gap'] = min_gap
                run_config_dict['subtitle_polishing']['min_reading_speed'] = min_reading_speed
                run_config_dict['subtitle_polishing']['max_reading_speed'] = max_reading_speed
                run_config_dict['subtitle_polishing']['preferred_reading_speed'] = preferred_reading_speed
                run_config_dict['subtitle_polishing']['merge_gap_threshold'] = merge_gap_threshold

                run_config_dict.setdefault('checkpointing', {})['enabled'] = checkpointing_enabled

                # Override working directory to be within the temp_dir for this run
                # This keeps outputs isolated and ensures they are cleaned up.
                run_working_dir = os.path.join(temp_dir, "processing_output")
                os.makedirs(run_working_dir, exist_ok=True)
                run_config_dict['paths']['working_directory'] = run_working_dir

                st.info(f"Run-specific working directory: {run_working_dir}")

                try:
                    # Instantiate AppConfigModel with the UI-derived and overridden settings
                    current_run_app_config = AppConfigModel(**run_config_dict)
                    st.info("Configuration for this run has been validated.")
                except Exception as e:
                    st.error(f"🚨 Configuration Error: Failed to create valid settings from UI inputs.")
                    st.code(f"{e}\n\n{traceback.format_exc()}", language="text")
                    # st.json(run_config_dict) # Optionally show the problematic dict
                    return # Stop processing

                log_output_placeholder = st.empty()
                # progress_bar = st.progress(0) # Progress bar needs more complex integration

                try:
                    st.info(f"Starting subtitle generation pipeline for '{uploaded_file.name}'...")
                    # This is a blocking call.
                    # The `execute_pipeline` function in `src/main.py` should handle its own logging.
                    # We will try to display the log file generated by `src/main.py` after execution.

                    main_execute_pipeline(app_config=current_run_app_config) # Direct call

                    # Define where the final SRT file should be, based on the run's config
                    final_srt_path = os.path.join(
                        current_run_app_config.paths.working_directory,
                        current_run_app_config.paths.final_srt_filename
                    )

                    if os.path.exists(final_srt_path):
                        # progress_bar.progress(100) # Simple completion indication
                        st.success("🎉 Subtitle generation complete!")

                        srt_content = ""
                        with open(final_srt_path, "r", encoding="utf-8") as f_srt:
                            srt_content = f_srt.read()

                        st.download_button(
                            label="📥 Download FINAL Subtitles (.srt)",
                            data=srt_content,
                            file_name=Path(uploaded_file.name).stem + "_subtitles.srt", # More descriptive name
                            mime="text/srt",
                        )

                        with st.expander("📜 SRT Preview (first 20 lines)", expanded=False):
                            st.text("".join(srt_content.splitlines(keepends=True)[:20]))
                    else:
                        st.error("Processing finished, but the final SRT file was not found. Check logs for details.")

                except Exception as e_pipeline:
                    st.error(f"🚨 An error occurred during subtitle generation pipeline: {e_pipeline}")
                    st.code(f"{traceback.format_exc()}", language="text")

                finally:
                    # Attempt to display log file content, regardless of success or failure of pipeline
                    log_file_path_ui = os.path.join(
                        current_run_app_config.paths.working_directory,
                        current_run_app_config.paths.log_filename
                    )
                    if os.path.exists(log_file_path_ui):
                        logging.info(f"Attempting to display log file: {log_file_path_ui}")
                        with open(log_file_path_ui, 'r', encoding='utf-8') as log_f:
                            log_content = log_f.read()
                            with st.expander("📄 View Full Log", expanded=False):
                                st.text_area("Log Output", value=log_content, height=400, key="log_area_final")
                    else:
                        st.warning(f"Log file not found at {log_file_path_ui}")

            st.info("Processing finished. You can now upload another video or adjust settings.")

# Add some footer or information
st.markdown("---")
st.markdown("Developed for AVT Workshop - Advanced Subtitling with AI.")
st.markdown("Considerations for production: error handling, real-time progress, advanced configuration options, etc.")

if execute_pipeline is None:
    st.error("🚨 Critical Error: The main processing pipeline (`execute_pipeline` from `src.main`) could not be imported. The application cannot function. Please check the server logs or `src/main.py` structure.")
