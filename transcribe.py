import time

from transformers import pipeline
import torch
from jiwer import wer
from deepgram import  DeepgramClient, PrerecordedOptions, FileSource

import asyncio


def transcribe_with_whisper(audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device, chunk_length_s=30,
                    generate_kwargs={"language": "<|en|>", "task": "transcribe"}, )

    start_time = time.time()
    prediction = pipe(audio_path, batch_size=8)
    end_time = time.time()

    transcription = prediction["text"]
    latency = end_time - start_time

    return transcription, latency


async def transcribe_with_deepgram(api_key, audio_path):
    dg_client = DeepgramClient(api_key)

    # open the audio file
    with open(audio_path, 'rb') as audio:
        buffer_data = audio.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    # start transcription and measure latency
    start_time = time.time()
    options = PrerecordedOptions(
        smart_format=True,
        summarize="v2",
    )
    response = dg_client.listen.rest.v("1").transcribe_file(payload, options)
    end_time = time.time()

    transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
    latency = end_time - start_time

    return transcription, latency


def calculate_wer(ground_truth_path, transcription):
    with open(ground_truth_path, 'r') as file:
        ground_truth = file.read()
    return wer(ground_truth, transcription)


def main():
    audio_path = "input_audio.wav"
    ground_truth_path = "ground_truth.txt"

    # here you can choose model (Whisper or Deepgram)
    use_whisper = False

    if use_whisper:
        print('Starting transcription with whisper')
        transcription, latency = transcribe_with_whisper(audio_path)
    else:
        print('Starting transcription with deepgram')
        api_key = "1a038da5010c1428b764c76d01a6c83a93054d67"
        transcription, latency = asyncio.run(transcribe_with_deepgram(api_key, audio_path))

    with open("transcription.txt", 'w') as file:
        file.write(transcription)

    wer_score = calculate_wer(ground_truth_path, transcription)

    with open("metrics.log", 'w') as log_file:
        log_file.write(f"Method used: {'whisper' if use_whisper else 'deepgram'}\n")
        log_file.write(f"Latency: {latency} seconds\n")
        log_file.write(f"Word Error Rate (WER): {wer_score}\n")


if __name__ == "__main__":
    main()
