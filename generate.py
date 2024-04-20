from transformers import AutoProcessor, BarkModel
from scipy.io import wavfile

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
# model.to("cuda")  # This line moves the model to GPU. Uncomment if using GPU.

def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset=preset)
    # Uncomment the following lines to move inputs to GPU if using GPU.
    # for k, v in inputs.items():
    #     inputs[k] = v.to("cuda")
    audio_array = model.generate(**inputs)
    # Uncomment the next line to move the output back to CPU from GPU if used.
    # audio_array = audio_array.cpu()
    audio_array = audio_array.numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    wavfile.write(output, rate=sample_rate, data=audio_array)

generate_audio(text="hi, welcome to my blog reader",
               preset="v2/en_speaker_2",
               output="output.wav")
