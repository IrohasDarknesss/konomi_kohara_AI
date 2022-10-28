import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import torch
from IPython.display import Audio
from scipy.io.wavfile import read, write
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "./model/checkpoint_2300"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = './model/waveglow_256channels_universal_v5.pt'
wg = torch.load(waveglow_path)['model']
wg.cuda().eval().half()
for k in wg.convinv:
    k.float()
denoiser = Denoiser(wg)

text = "darenoyrushiwoetebokuwomiteiru, kyoukenmega."
sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))

with torch.no_grad():
    audio = wg.infer(mel_outputs_postnet, sigma=0.666)
ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

audio_numpy = audio[0].data.cpu().numpy()
rate = 22050
from IPython.display import Audio
from scipy.io.wavfile import read, write
Audio(audio_numpy, rate=rate)
write("./voicesample/kohara.wav", rate, audio_numpy)