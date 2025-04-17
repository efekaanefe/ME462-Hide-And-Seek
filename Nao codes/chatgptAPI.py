# -*- coding: utf-8 -*-

import requests
from naoqi import ALProxy

nao_ip = "192.168.0.202"

def ask_ollama_short_english(prompt):
    full_prompt = (
        "You are a polite robot that gives short and simple answers in English. "
        "Please respond in no more than 2 sentences.\n"
        "Question: " + prompt
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": full_prompt,
            "stream": False
        }
    )

    reply = response.json()["response"]
    return reply.strip()

tts = ALProxy("ALTextToSpeech", nao_ip, 9559)
tts.setLanguage("English")
tts.setParameter("pitchShift", 1.2)

while True:
    user_input = raw_input("You: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break

    answer = ask_ollama_short_english(user_input)
    print("🤖:", answer)
    tts.say(str(answer))

