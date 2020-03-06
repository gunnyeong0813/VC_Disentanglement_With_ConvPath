import os
import time

from speech_tools import *

#dataset = 'vcc2018'


speaker_list =  ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2']
data_dir = os.path.join('.')
exp_dir = os.path.join('processed')
store_dir =  os.path.join('processed_eval')
#start_time = time.time()

sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128
for speaker in speaker_list:
    flist = sorted(glob.glob(data_dir+'/'+ speaker + '/*.wav'))
    for file in flist:
        train_A_dir = os.path.join(data_dir, speaker)
        #train_B_dir = os.path.join(data_dir, trg_speaker)
        exp_A_dir = os.path.join(exp_dir, speaker)
        #exp_B_dir = os.path.join(exp_dir, trg_speaker)
        exp_store_dir =  os.path.join(store_dir, speaker)
        os.makedirs(exp_store_dir, exist_ok=True)
        #os.makedirs(exp_B_dir, exist_ok=True)
        print('Loading {} features...'.format(speaker))
        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
        os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
        print('Extracting for MCD features...')
        
        wav, _ = librosa.load(file, sr=22050, mono=True)
        wav = librosa.util.normalize(wav, norm=np.inf, axis=None)      
        wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
        f0, timeaxis, sp, ap,mc = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
        #coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
               
        #coded_sp_transposed = coded_sp.T
        #coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
        print('Saving {} data...'.format(speaker))
        save_pickle(os.path.join(exp_store_dir, '{}.p'.format(file.split('/')[-1].split('.')[0])),
                    (mc, ap, f0))



end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')

print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
