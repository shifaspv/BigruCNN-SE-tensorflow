import os
import threading
import logging
import logging
import pdb
import numpy as np
import tensorflow as tf
import lib.util as util
from lib.util import load_wav
from lib.precision import _FLOATX
from numpy.lib import stride_tricks
log10_fac = 1 / np.log(10)
from numpy import inf
def wav_to_float(x):
    '''try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.finfo(x.dtype).min
    print(np.min(x), np.max(x))
    x = x.astype("float64", casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.'''
    x = x.astype(_FLOATX.as_numpy_dtype()) #, casting='safe')  
    return x




def get_subsequence_with_speech_indices(full_sequence, min_length, sample_rate, silence_threshold=0.1):
    signal_magnitude = np.abs(full_sequence)

    chunk_length = max(1, int(sample_rate*0.005)) # 5 milliseconds

    chunks_energies = []
    for i in range(0, len(signal_magnitude), chunk_length):
        chunks_energies.append(np.mean(signal_magnitude[i:i + chunk_length]))

    threshold = np.max(chunks_energies) * silence_threshold

    onset_chunk_i = 0
    for i in range(0, len(chunks_energies)):
        if chunks_energies[i] >= threshold:
            onset_chunk_i = i
            break

    termination_chunk_i = len(chunks_energies)
    for i in range(len(chunks_energies) - 1, 0, -1):
        if chunks_energies[i] >= threshold:
            termination_chunk_i = i
            break

    if (termination_chunk_i - onset_chunk_i)*chunk_length >= min_length: # Then pad, else ignore
        num_pad_chunks = 4
        onset_chunk_i = np.max((0, onset_chunk_i - num_pad_chunks))
        termination_chunk_i = np.min((len(chunks_energies), termination_chunk_i + num_pad_chunks))

    return [onset_chunk_i*chunk_length, (termination_chunk_i+1)*chunk_length]


def extract_subsequence_with_speech(clean_audio, noisy_audio, min_length, fs, silence_threshold=0.1):

    indices = get_subsequence_with_speech_indices(clean_audio, min_length, fs, silence_threshold)

    if indices[0] == indices[1]:
        return None, None, None
    else:    
        return clean_audio[indices[0]:indices[1]], noisy_audio[indices[0]:indices[1]]


def read_filelist(file_list):
    fid = open(file_list, 'r')
    lines = fid.readlines()
    fid.close()
    
    filenames = []
    for filename in lines:
        filenames.append( filename.rstrip() )

    return filenames



def load_noisy_audio_label_and_speaker_id(filename, test_noisy_audio_dir, audio_ext, sample_rate):

    noisy_audio_fullpathname = os.path.join(test_noisy_audio_dir, filename + audio_ext)
    noisy_audio = load_wav(noisy_audio_fullpathname, sample_rate)

   
    return noisy_audio


def load_clean_noisy_audio_and_label(filename, clean_audio_dir, noisy_audio_dir, audio_ext='.wav', sample_rate=16000):
    '''Reads an audio file and the corresponding phonetic unit labels.'''
    

    clean_audio_fullpathname = os.path.join(clean_audio_dir, filename.rstrip() + audio_ext)
    clean_audio = load_wav(clean_audio_fullpathname, sample_rate)
        
    n_audio_samples = len(clean_audio) 

    noisy_audio_fullpathname = os.path.join(noisy_audio_dir, filename.rstrip() + audio_ext)
    noisy_audio = load_wav(noisy_audio_fullpathname, sample_rate)

    assert(n_audio_samples == len(noisy_audio))
 
    return clean_audio, noisy_audio

def align_audio_samples_to_label_samples(filename, audio_dir, audio_ext='.wav',sample_rate=16000):

    audio_fullpathname = os.path.join(audio_dir, filename.rstrip() + audio_ext)
    audio = load_wav(audio_fullpathname, sample_rate)
        
    return audio

class AudioConditionsReader(object):
    '''Generic background audio and label reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord, 
                 file_list,
                 clean_audio_dir,
                 noisy_audio_dir,
                 audio_ext,
                 sample_rate,
                 regain,
                 num_input_frames, 
                 frame_size,
                 frame_shift,
                 masker_length,
                 batch_size=1,                 
                 queue_size=64, 
                 permute_segments=False):

        self.coord = coord
        self.file_list = file_list
        self.clean_audio_dir = clean_audio_dir
        self.noisy_audio_dir = noisy_audio_dir
        self.audio_ext = audio_ext
        self.sample_rate = sample_rate
        self.regain = regain
        self.batch_size = batch_size 
        self.num_input_frames=num_input_frames
        self.frame_size=int(frame_size*sample_rate)
        self.frame_shift=int(frame_shift*sample_rate)
        self.masker_length=masker_length
        self.permute_segments = permute_segments 
        self.EFFT=int((self.masker_length*2))
        self.dim_out=int(self.EFFT/2+1)
#        if input_length is not None:
#            self.input_length = input_length
#            self.target_length = input_length - (receptive_field - 1)
#        elif target_length is not None:
#            self.input_length = int(target_length + (receptive_field - 1))   
#            self.target_length = target_length 
#        else:
#            self.input_length = None
#            self.target_length = None
        self.input_length=self.num_input_frames
        self.target_length=self.input_length


        self.logger = logging.getLogger("warning_logger")

        self.audio_input_placeholder = tf.placeholder(dtype=_FLOATX, shape=(batch_size, self.dim_out, self.input_length, 1)) 
        self.audio_input_queue = tf.FIFOQueue(queue_size, dtypes=[_FLOATX], shapes=[(batch_size,self.dim_out, self.input_length, 1)])
        self.audio_input_enqueue_op = self.audio_input_queue.enqueue(self.audio_input_placeholder)

        self.audio_output1_placeholder = tf.placeholder(dtype=_FLOATX, shape=(batch_size,self.dim_out, self.target_length, 1)) 
        self.audio_output1_queue = tf.FIFOQueue(queue_size, dtypes=[_FLOATX], shapes=[(batch_size,self.dim_out,self.target_length, 1)])
        self.audio_output1_enqueue_op = self.audio_output1_queue.enqueue(self.audio_output1_placeholder)

#        self.audio_output2_placeholder = tf.placeholder(dtype=_FLOATX, shape=(batch_size, 1, self.input_length, 1)) 
#        self.audio_output2_queue = tf.FIFOQueue(queue_size, dtypes=[_FLOATX], shapes=[(batch_size, 1, self.input_length, 1)])
#        self.audio_output2_enqueue_op = self.audio_output2_queue.enqueue(self.audio_output2_placeholder)

#        pdb.set_trace()

        self.filenames = read_filelist(file_list)

        self.n_files = len(self.filenames)

        self.indices_list = self.find_segment_indices()

        self.n_segments = len(self.indices_list) 
    
        self.perm_indices = np.arange(self.n_segments) 

        self.reset()

    def reset(self):
        self.enqueue_finished = False
        self.dequeue_finished = False
        self.n_enqueued = 0
        self.n_dequeued = 0

        if self.permute_segments: 
            np.random.shuffle(self.perm_indices) 

    
    def find_segment_indices(self):

        indices_list = []
        
        for i, filename in enumerate(self.filenames):
            #clean_audio_fullpathname = os.path.join(self.clean_audio_dir, filename.rstrip() + self.audio_ext)
            #audio = load_wav(clean_audio_fullpathname, self.sample_rate)
            #clean_audio, noisy_audio, label = load_clean_noisy_audio_and_label(filename, self.clean_audio_dir, self.noisy_audio_dir, 
            #                       self.lc_enabled, self.label_dir, self.label_dim, self.audio_ext, self.label_ext, self.sample_rate,
            #                       self.frame_length, self.frame_shift, self.lc_context_length) 

            audio = align_audio_samples_to_label_samples(filename, self.clean_audio_dir, self.audio_ext, self.sample_rate)

#            if self.silence_threshold > 0:
#                # Remove silence 
#                indices = get_subsequence_with_speech_indices(audio, self.input_length, self.sample_rate, self.silence_threshold)
#                if indices[0] == indices[1]:
#                    audio = None
#                else: 
#                    audio = audio[indices[0]:indices[1]]

            if (audio is None):
                self.logger.warning("Warning: {} was ignored as it contains only "
                      "silence. Consider decreasing the silence threshold.".format(filename)) 
                continue 

            regain_factor = self.regain / util.rms(audio)

            # Cut samples into pieces of size input with an overlap of input_length-target_length
            n_audio_samples = len(audio)
            
            num_itr=int((n_audio_samples-self.frame_size)/self.frame_shift-self.num_input_frames)
            
            if num_itr< self.batch_size:
                continue

#            if self.input_length is None:
#                self.input_length = n_audio_samples
#                self.target_length = self.target_length  
#                from_index = 0
#                to_index = n_audio_samples 
#                indices_list.append((filename, from_index, to_index, regain_factor)) 
#            else:
#                from_index = 0
#                to_index = self.input_length
#                while n_audio_samples - from_index >= self.input_length+ (0.5*self.target_length):
#                    if to_index > n_audio_samples:
#                        from_index = n_audio_samples - self.input_length
#                        if from_index < 0:
#                            break 
#                        to_index = n_audio_samples 
#                     
#                    indices_list.append((filename, num_itr, regain_factor))
#                    from_index += self.target_length
#                    to_index += self.target_length
            indices_list.append((filename, num_itr, regain_factor))
        return indices_list
      

    def check_for_elements_and_increment(self): 
   
        if self.enqueue_finished and (self.n_enqueued == self.n_dequeued):
            return False
        else:
            self.n_dequeued += 1 
            return True


    def dequeue(self): 
        audio_input = self.audio_input_queue.dequeue()
        audio_output1 = self.audio_output1_queue.dequeue()
#        audio_output2 = self.audio_output2_queue.dequeue()

#        audio_shape = tf.shape(audio_input)    

        return audio_input, audio_output1


    def enqueue_thread(self, sess):
      
        for j in range(0, self.n_segments-self.batch_size, self.batch_size): 
            
            input_batch = np.zeros((self.batch_size, self.dim_out, self.input_length, 1), dtype=_FLOATX.as_numpy_dtype())
            output1_batch = np.zeros((self.batch_size, self.dim_out, self.target_length, 1), dtype=_FLOATX.as_numpy_dtype())
            for k in range(self.batch_size): 
                    i = self.perm_indices[j + k]
            
                    filename, num_itr, regain_factor= self.indices_list[i]
    
                    clean_audio, noisy_audio = load_clean_noisy_audio_and_label(filename, self.clean_audio_dir, self.noisy_audio_dir, self.audio_ext, self.sample_rate)  
    
    #                if self.silence_threshold:
    #                    # Remove silence from the beginning and end of a utterance
    #                    clean_audio, noisy_audio= extract_subsequence_with_speech(clean_audio, noisy_audio,
    #                                                                   self.input_length, self.sample_rate, self.silence_threshold)  
    
#                    noise = noisy_audio - clean_audio
#                    pdb.set_trace() 
                    noise_signal=noisy_audio-clean_audio
                    clean_audio_segment_regained = clean_audio * regain_factor
                    noise_segment_regained = noise_signal * regain_factor
                    noisy_audio_segment_regained=clean_audio_segment_regained+noise_segment_regained
                    clean_audio_segments = stride_tricks.as_strided(clean_audio_segment_regained,
                                                                    shape=(num_itr,self.num_input_frames,self.frame_size),
                                                                    strides=(
                                                                            clean_audio.strides[0]*self.frame_shift,
                                                                            clean_audio.strides[0]*self.frame_shift,
                                                                            clean_audio.strides[0]))
    
                    noisy_audio_segments = stride_tricks.as_strided(noisy_audio_segment_regained,
                                                                    shape=(num_itr,self.num_input_frames,self.frame_size),
                                                                    strides=(
                                                                            noisy_audio.strides[0]*self.frame_shift,
                                                                            noisy_audio.strides[0]*self.frame_shift,
                                                                            noisy_audio.strides[0]))
                    # pdb.set_trace()
                    batch_index=np.random.randint(num_itr, size=1)
                    clean_audio_segments=clean_audio_segments[batch_index,:,:]
                    noisy_audio_segments=noisy_audio_segments[batch_index,:,:]
                    clean_segment=10*np.log10(10**2*np.abs(np.fft.rfft(clean_audio_segments, n=self.EFFT))**2+1)
                    noisy_segment=10*np.log10(10**2*np.abs(np.fft.rfft(noisy_audio_segments, n=self.EFFT))**2+1)
#                    clean_segment[clean_segment == -inf] = 0
#                    noisy_segment[noisy_segment== -inf] =0
    #                noise_segment = noise[from_index:to_index]   # bug fix noisy_audio -> noise
    
    #                input_segment = clean_audio_segment_regained + noise_segment_regained
    
#                    output1_segment = clean_audio_segment_regained
    
      
    #                if self.noise_only_percent > 0:
    #                    if np.random.uniform(0, 1) <= self.noise_only_percent:
    #                        input_segment = noise_segment_regained
    #                        output1_segment = np.array([0]*input_segment.shape[0], dtype=_FLOATX.as_numpy_dtype()) 
                    
                    #print(filename, from_index, to_index, clean_audio.shape)
                                      
                    input_batch[k, :, :, 0] = np.transpose(noisy_segment,[0,2,1])
                    output1_batch[k, :, :, 0] = np.transpose(clean_segment,[0,2,1])
#            pdb.set_trace() 
#            data_mean=np.mean(output1_batch,axis=0)
#            data_std=np.var(output1_batch,axis=0)
#            input_batch -=data_mean
#            input_batch /=data_std
#            output1_batch -=data_mean
#            output1_batch /=data_std
            sess.run([self.audio_input_enqueue_op, self.audio_output1_enqueue_op], 
                     feed_dict={self.audio_input_placeholder: input_batch, self.audio_output1_placeholder: output1_batch})
                          
            self.n_enqueued += 1
                       
        self.enqueue_finished = True
                    
                    
    def start_enqueue_thread(self, sess):
        thread = threading.Thread(target=self.enqueue_thread, args=(sess, ))
        thread.start()
        return thread
