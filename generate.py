from transformers import AutoProcessor, BarkModel
from scipy.io import wavfile

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cuda")  # Remove this line if you don't have a CUDA-compatible GPU

def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset=preset)
    for k, v in inputs.items():
        inputs[k] = v.to("cuda")  # Remove this line if you don't have a CUDA-compatible GPU
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    wavfile.write(output, rate=sample_rate, data=audio_array)

generate_audio(text="hi, welcome to my blog reader",
               preset="v2/en_speaker_2",
               output="output.wav")
