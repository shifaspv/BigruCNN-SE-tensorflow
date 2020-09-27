from __future__ import print_function

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.

import os
import logging
import tensorflow as tf
from model3 import gruCNN_SE
from lib.optimizers import get_learning_rate, get_optimizer
from lib.audio_conditions_io import AudioConditionsReader, load_noisy_audio_label_and_speaker_id
from lib.model_io import get_configuration, setup_logger
from lib.model_io import restore_variables
from lib.util import load_wav, write_wav, compute_receptive_field_length
import pdb

cfg, _, _= get_configuration('generation')

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["CUDA_VISIBLE_DEVICES"]
cfg["batch_size"]=1

coord = tf.train.Coordinator()

# Create the network
gruCNN_SE = gruCNN_SE(cfg)

# gruCNN_SE.define_generation_computations()
#gruCNN_SE.define_generation_computations([1,cfg['fft_bin_size']/2+1, cfg['num_input_frames'],1])

#pdb.set_trace()
# Recover the parameters of the model
# cfg['model_id']=3


# Generate waveform
# pdb.set_trace()
#cfg['test_clean_audio_dir'] ='../data/NSDTSEA/clean_testset_wav'
sess = tf.Session()

noisy_audio= load_noisy_audio_label_and_speaker_id(cfg['noisy_speech_filename'], cfg['test_noisy_audio_dir'], cfg['audio_ext'], 
                                                        cfg['sample_rate'])
clean_audio= load_noisy_audio_label_and_speaker_id(cfg['noisy_speech_filename'], cfg['test_clean_audio_dir'], cfg['audio_ext'], 
                                                        cfg['sample_rate'])    

#pdb.set_trace()
clean_audio, noise = gruCNN_SE.generation(sess, noisy_audio, clean_audio, cfg)   

wav_out_path = os.path.join(cfg['base_dir'], cfg['output_dir'], str(cfg['model_id']))
if not os.path.exists(wav_out_path):
  os.makedirs(wav_out_path)

clean_audio_out_fullpathname = os.path.join(wav_out_path, cfg['output_clean_speech_filename'] + cfg['audio_ext']) 
print(clean_audio_out_fullpathname, clean_audio.shape, clean_audio.dtype, cfg['sample_rate']) 
write_wav(clean_audio, clean_audio_out_fullpathname, cfg['sample_rate'])

noise_out_fullpathname = os.path.join(wav_out_path, cfg['output_noise_filename'] + cfg['audio_ext']) 
#write_wav(noise, noise_out_fullpathname, cfg['sample_rate'])
#pdb.set_trace()

#noise_out_fullpathname = os.path.join(wav_out_path, cfg['output_noise_filename'] + cfg['audio_ext']) 
#write_wav(noise, noise_out_fullpathname, cfg['sample_rate'])
#pdb.set_trace()
