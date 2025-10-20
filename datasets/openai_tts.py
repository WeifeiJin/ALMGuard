import os
import requests
import json

api_key = "your_api_key"
api_base = "your_base_url"

voice_dict = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

def openai_tts(prompt, voice_id=0, speed=1.0):

    url = f"{api_base}/audio/speech"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "tts-1",
        "voice": voice_dict[voice_id],
        "input": prompt,
        "speed": speed
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Calling TTS failed: {e}")
        return

    audio_data = response.content
    return audio_data



if __name__ == "__main__":
    text = "Hello, this is an AI-generated voice from SmartTrot."
    output_file = "output_audio.wav"
    audio = openai_tts(text, voice_id=2, speed=1.2)

    with open(output_file, "wb") as f:
        f.write(audio)