# AVT Subtitler Pro

This project is an enterprise-grade, AI-powered pipeline for generating professional Indonesian subtitles from English videos.

## Project Structure

For the application to run correctly, your project directory should be structured as follows. The main script assumes it can import from `config.py` (which is in the root, and `src/main.py` can import from it if the parent of `src` is in PYTHONPATH, which `python -m` handles).

```
/avt_subtitler_project/
|-- src/
|   |-- main.py          # The main application script
|   |-- __init__.py      # Make src a package
|-- config.py            # The Pydantic configuration models (root)
|-- config.yaml          # The user-facing configuration file (root)
|-- app.py               # (Optional) Streamlit UI (root)
|-- requirements.txt     # Project dependencies (root)
|-- assets/              # Example directory for input files (relative to where you run from, or use absolute paths in config)
|   |-- input.mp4
|   |-- mapping.json
|-- output/              # Example directory for all generated files (defined in config.yaml)
```

## Installation

1.  Ensure you have Python 3.8+ installed.
2.  Clone the repository (if applicable).
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Set up your `config.yaml`:
    *   Copy `config.yaml.example` to `config.yaml` (if an example is provided).
    *   Edit `config.yaml` to set correct paths for `dataset_path` (for assets like `mapping.json`) and `working_directory` (for outputs).
    *   Provide your Hugging Face token in `config.yaml` if using speaker diarization with PyAnnote models (`content_generation.diarization.hf_token`).

## Execution

To run the main subtitle generation pipeline:

Navigate to the directory *containing* the `avt_subtitler_project` folder (i.e., the parent directory of `avt_subtitler_project`). Then, execute the script as a module:

```bash
python -m avt_subtitler_project.src.main
```

This command runs the `main` function within `src/main.py` using the default `config.yaml` located in the root of `avt_subtitler_project`.

To run the Streamlit web application (if available):

Navigate to the root of the `avt_subtitler_project` folder and run:
```bash
streamlit run app.py
```
This will ensure all Python imports resolve correctly.
