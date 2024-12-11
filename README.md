# Speech-to-Text Transcription Tool

This repository contains a Python-based Speech-to-Text (STT) transcription tool that supports transcription using either **OpenAI Whisper** or **Deepgram's API**. It also evaluates transcription accuracy using the **Word Error Rate (WER)** metric.

---

## Features

- **Transcription with OpenAI Whisper**:
  - Uses Hugging Face's `transformers` library.
  - Supports chunked audio processing.
  - Configurable for different languages and tasks.

- **Transcription with Deepgram**:
  - Utilizes Deepgram’s REST API for real-time transcription.
  - Includes advanced options like smart formatting and summarization.

- **Performance Metrics**:
  - Logs transcription latency.
  - Calculates Word Error Rate (WER) for accuracy evaluation.

---

## Requirements

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for Whisper).

### Install Dependencies

Install the required libraries:
```bash
pip install -r requirements.txt
```

##Usage
1. Prepare Input Files
Audio File: Place your audio file in the project directory and name it input_audio.wav (must be in WAV format).
Ground Truth File: Create a file named ground_truth.txt containing the reference text(ground truth) for WER calculation.
2. Select Transcription Method
In the main() function of main.py, set the use_whisper variable:

True: Use Whisper for transcription.
False: Use Deepgram for transcription.
3. Run the transcribe.py file
4. Output Files
transcription.txt: The transcribed text.
metrics.log: Contains latency and WER metrics.
## License
This project is open-source and distributed under the MIT License.