#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Audio
from scipy.io.wavfile import write as write_wav

from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic


# In[ ]:


semantic_path = "semantic_output/pytorch_model.bin" # set to None if you don't want to use finetuned semantic
coarse_path = "coarse_output/pytorch_model.bin" # set to None if you don't want to use finetuned coarse
fine_path = "fine_output/pytorch_model.bin" # set to None if you don't want to use finetuned fine
use_rvc = True # Set to False to use bark without RVC
rvc_name = 'model'
rvc_path = f"trained/ian/{rvc_name}.pth"
index_path = f"trained/ian/{rvc_name}.index"
device="cuda:0"
is_half=True


# In[ ]:


# download and load all models
preload_models(
    text_use_gpu=True,
    text_use_small=False,
    text_model_path=semantic_path,
    coarse_use_gpu=True,
    coarse_use_small=False,
    coarse_model_path=coarse_path,
    fine_use_gpu=True,
    fine_use_small=False,
    fine_model_path=fine_path,
    codec_use_gpu=True,
    force_reload=False,
    path="models"
)

if use_rvc:
    from rvc_infer import get_vc, vc_single
    get_vc(rvc_path, device, is_half)


# In[ ]:


# simple generation
text_prompt = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
voice_name = "speaker_0" # use your custom voice name here if you have on

filepath = "output/audio.wav"
audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
write_wav(filepath, SAMPLE_RATE, audio_array)

if use_rvc:
    index_rate = 0.75
    f0up_key = -6
    filter_radius = 3
    rms_mix_rate = 0.25
    protect = 0.33
    resample_sr = SAMPLE_RATE
    f0method = "harvest" #harvest or pm
    try:
        audio_array = vc_single(0,filepath,f0up_key,None,f0method,index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
    except:
        audio_array = vc_single(0,filepath,f0up_key,None,'pm',index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
    write_wav(filepath, SAMPLE_RATE, audio_array)

Audio(audio_array, rate=SAMPLE_RATE)


# In[ ]:


# generation with more control
text_prompt = "Hello, my name is Serpy. And, uh — and I like pizza. [laughs]"
voice_name = "speaker_0" # use your custom voice name here if you have on

filepath = "output/audio.wav"

x_semantic = generate_text_semantic(
    text_prompt,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)

x_coarse_gen = generate_coarse(
    x_semantic,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)
x_fine_gen = generate_fine(
    x_coarse_gen,
    history_prompt=voice_name,
    temp=0.5,
)
audio_array = codec_decode(x_fine_gen)
write_wav(filepath, SAMPLE_RATE, audio_array)

if use_rvc:
    index_rate = 0.75
    f0up_key = -6
    filter_radius = 3
    rms_mix_rate = 0.25
    protect = 0.33
    resample_sr = SAMPLE_RATE
    f0method = "harvest" #harvest or pm
    try:
        audio_array = vc_single(0,filepath,f0up_key,None,f0method,index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
    except:
        audio_array = vc_single(0,filepath,f0up_key,None,'pm',index_path,index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
    write_wav(filepath, SAMPLE_RATE, audio_array)

Audio(audio_array, rate=SAMPLE_RATE)


# In[ ]:




