import os
import librosa
import soundfile as sf

# Path
input_path = './16bit/'
output_path = './32bit/'

# Making Print Folder
os.makedirs(output_path, exist_ok=True)

# convert 16bit 
filenames = os.listdir(input_path)
for filename in filenames:
    print(input_path+filename)
    y, sr = librosa.core.load(input_path+filename, sr=22050, mono=True)
    sf.write(output_path+filename, y, sr, subtype="PCM_16")