from audiosense import AudioSense

audio = r"C:\Users\ASUS\Desktop\Major Project\AudioSense_old\audio\audio.wav"

model = AudioSense(device="auto",language="en")
res = model.predict(audio)
print(res)