#!/usr/bin/env python
# coding: utf-8
import os
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav

from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from LearningBuddyCommon.integration.integration_utils import get_config_file

# In[ ]:

config = get_config_file("input/config.json")
semantic_path = "semantic_output/pytorch_model.bin" # set to None if you don't want to use finetuned semantic
coarse_path = "coarse_output/pytorch_model.bin" # set to None if you don't want to use finetuned coarse
fine_path = "fine_output/pytorch_model.bin" # set to None if you don't want to use finetuned fine
use_rvc = config["use_rvc"] # Set to False to use bark without RVC
rvc_path = config.get("model_path")
index_path = config.get("model_index_path")
device="cuda:0"
is_half=True


import nltk  # we'll use this to split into sentences
import numpy as np

OUTPUT_FOLDER="output"

# download and load all models
nltk.download('punkt')

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


# generation with more control
text_prompt = config.get("prompt")
voice_name = config.get("base_voice") # use your custom voice name here if you have on

filepath = os.path.join(OUTPUT_FOLDER, config.get("output_audio_path"))

sentences = nltk.sent_tokenize(text_prompt)
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
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
        temp=0.45,
    )
    audio_array = codec_decode(x_fine_gen)
    pieces += [audio_array, silence.copy()]
result = np.concatenate(pieces)    
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




