import json
import os

import numpy as np

from datetime import datetime
from importlib import import_module
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from speech_tools import *
import argparse
import soundfile
import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle

from solver_multi_decoder import Solver

from scipy.io.wavfile import write

import h5py 
import os 
import time

def get_model(speaker_num, model_path='./model.pkl-243000'):
    solver = Solver()
    print(speaker_num)
    solver.load_model(str(model_path),speaker_num=speaker_num)
    return solver

def label_generate(num,batch):
        label = []
        for i in range(batch):
            label.append(num)
        return torch.Tensor(label)
        
def label2onehot(labels, batch_size):
        labels = torch.Tensor(labels)
        out = torch.zeros(batch_size, 4)
        out[np.arange(batch_size), labels.long()] = 1
        return out

def label_generate(num,batch):
        label = []
        for i in range(batch):
            label.append(num)
        return torch.Tensor(label)

def conversion(model_dir, source, target):
    for source in range(4):
        for target in range(4):
          if source!=target:
            speaker_list = sorted(os.listdir("processed"))
            src_speaker = speaker_list[source]
            trg_speaker = speaker_list[target]
            
            data_dir = os.path.join('data')
            exp_dir = os.path.join('processed')
            test_dir = os.path.join('processed_eval')
            eval_A_dir = os.path.join(data_dir, 'speakers_test', src_speaker)
            eval_B_dir = os.path.join(data_dir, 'speakers_test', trg_speaker)
            exp_A_dir = os.path.join(exp_dir, src_speaker)
            exp_B_dir = os.path.join(exp_dir, trg_speaker)
            exp_test_dir =os.path.join(test_dir,src_speaker)
            validation_A_output_dir = os.path.join('converted_voices', 'converted_{}_to_{}'.format(src_speaker, trg_speaker))
         
            os.makedirs(validation_A_output_dir, exist_ok=True)
           
            sampling_rate = 22050
            num_mcep = 36
            frame_period = 5.0
            n_frames = 128

            print('Loading cached data...')
            coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
                os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
            coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
                os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))


            
            count=0
            if src_speaker ==trg_speaker:
                    for i in range(4):
                        if src_speaker == speaker_list[count]:
                            src_num = str(count)
                            trg_num = str(count)
                        count = count + 1
            else:
                    for i in range(4):
                        if src_speaker == speaker_list[count]:
                            src_num = str(count)
                        elif trg_speaker == speaker_list[count]:
                            trg_num = str(count)
                        count = count + 1
            print("target num ",trg_num)
            print("source num ",src_num)

            solver = get_model(trg_num,model_dir)
         
            device = torch.device("cuda")
          
            
            src_c = label_generate(int(src_num),1)
            trg_c = label_generate(int(trg_num),1)
            src_c = label2onehot(src_c,1).to(device, dtype=torch.float)
            trg_c = label2onehot(trg_c,1).to(device, dtype=torch.float)
            
            flist = sorted(glob.glob(eval_A_dir + '/*.wav'))    
            for file in flist:
                       
                        sp, ap, f0 = load_pickle(os.path.join(exp_test_dir, '{}.p'.format(file.split('/')[-1].split('.')[0])))
                        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                                        mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
                      
                        coded_sp_transposed =  sp.T
                        
                        coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                        coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
                        coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
                        coded_sp_norm = torch.from_numpy(coded_sp_norm).to(device, dtype=torch.float)
                      
                        coded_sp_converted_norm = solver.test_step(coded_sp_norm, src_c,trg_c,int(trg_num), gen=False)
                        coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)
                        coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)
                        if coded_sp_converted_norm.shape[1] > len(f0):
                        
                                coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
                        coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                        
                        coded_sp_converted = coded_sp_converted.T
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        
                       
                        
                        decoded_sp_converted = world_decode_mc(mc=coded_sp_converted, fs=sampling_rate)
                      
                        if coded_sp_converted_norm.shape[1] < len(f0):
                                f0_converted = f0_converted[:int(coded_sp_converted_norm.shape[1])]
                                ap = ap[:int(coded_sp_converted_norm.shape[1])]
                        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                                                frame_period=frame_period)
     
                        wav_transformed = np.nan_to_num(wav_transformed)
            
                        soundfile.write(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='Convert voices using pre-trained Model.')

      model_dir = './logdir/train/0719-0139-35-2019/'
      source_speaker = 'SF1'
      target_speaker = 'TM3'
  
      parser.add_argument('--model_dir', type=str, help='Directory for the pre-trained model.', default=model_dir)
      parser.add_argument('--source_speaker', type=str, help='source_speaker', default=source_speaker)
      parser.add_argument('--target_speaker', type=str, help='target_speaker', default=target_speaker)

      argv = parser.parse_args()
  
      model_dir = argv.model_dir
      source_speaker = argv.source_speaker
      target_speaker = argv.target_speaker
  
      conversion(model_dir = model_dir, source=source_speaker, target=target_speaker)
      