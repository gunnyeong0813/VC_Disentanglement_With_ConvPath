import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from model import Encoder
from model import Decoder
from model import Discriminator
from model import ACLayer
from model import ASRLayer
from model import SpeakerClassifier

from speech_tools import load_pickle, sample_train_data
import os
import math
from utils import reset_grad
from utils import grad_clip
from utils import cc
from matplotlib import pyplot as plt
import librosa
import time
import random

class Solver(object):
    def __init__(self, log_dir='./log/'):
        self.model_kept = []
        self.max_keep=100
        self.build_model()
        self.device = torch.device("cuda")

    def build_model(self):
        self.Encoder = cc(Encoder())
        self.Decoder = [cc(Decoder()) for _ in range(4)]
        self.ACLayer = cc(ACLayer())
        self.Discriminator= cc(Discriminator())
        self.ASRLayer = cc(ASRLayer())
        self.SpeakerClassifier = cc(SpeakerClassifier())
        ac_betas = (0.5,0.999)
        vae_betas = (0.9,0.999)
        ac_lr = 0.00005
        vae_lr = 0.001
        dis_lr = 0.002
        cls_betas = (0.5,0.999)
        asr_betas = (0.5,0.999)
        cls_lr = 0.0002
        asr_lr = 0.00001

        self.list_decoder = []

        for i in range(4):
            self.list_decoder+=list(self.Decoder[i].parameters())
        self.vae_params = list(self.Encoder.parameters())+self.list_decoder
        self.ac_optimizer = optim.Adam(self.ACLayer.parameters(), lr=ac_lr, betas=ac_betas)
        self.vae_optimizer = optim.Adam(self.vae_params, lr=vae_lr, betas=vae_betas)
        self.dis_optimizer = optim.Adam(self.Discriminator.parameters(), lr=dis_lr, betas=ac_betas)
        
        self.asr_optimizer = optim.Adam(self.ASRLayer.parameters(), lr=asr_lr, betas=asr_betas)
        self.cls_optimizer = optim.Adam(self.SpeakerClassifier.parameters(), lr=cls_lr, betas=cls_betas)
        

        
        

    def save_model(self, model_path, epoch,enc_only=True):
        all_model=dict()
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),
            }
        else:
            all_model['encoder'] = self.Encoder.state_dict()
            
            for i in range(4):
                model_name = 'decoder'+str(int(i+1))
                all_model[model_name] = self.Decoder[i].state_dict()
            
            all_model['aclayer'] = self.ACLayer.state_dict()
            
         
        new_model_path = os.path.join(model_path,'{}-{}'.format(model_path, epoch))
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path, speaker_num,enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            speaker_index = 'decoder'+str(int(speaker_num)+1)
            self.Decoder[int(speaker_num)].load_state_dict(all_model[speaker_index])
           
    def load_whole_model(self,model_path,enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])

            self.Decoder.load_state_dict(all_model['decoder'])
    
    def set_train(self):
        self.Encoder.train()
        for i in range(4):
          self.Decoder[i].train()
        
       
    def set_eval(self,trg_speaker_num):
        self.Encoder.eval()
        self.Decoder[trg_speaker_num].eval()
      
    def test_step(self, x, src_c,trg_c,trg_speaker_num, gen=False):
        self.set_eval(trg_speaker_num)
       
        en_mu,en_lv = self.Encoder(x,src_c)
        z = self.reparameterize(en_mu,en_lv)
        xt_mu,xt_lv = self.Decoder[trg_speaker_num](en_mu, trg_c)
        x_tilde = self.reparameterize(xt_mu,xt_lv)
        
        return xt_mu.data.cpu().numpy()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)
    
    def grad_reset(self):
        self.ac_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        self.asr_optimizer.zero_grad()
        self.cls_optimizer.zero_grad()
  
    def KLD_loss(self, mu,logvar):
        mu2 = torch.zeros_like(mu)
        logvar2 = torch.zeros_like(logvar)
        logvar = logvar.exp()
        logvar2 = logvar2.exp()
        
        mu_diff_sq = mu - mu2
        mu_diff_sq = mu_diff_sq.pow(2)
        
        dimwise_kld = .5 * (
            (logvar2 - logvar) + torch.div(logvar + mu_diff_sq, logvar2 + 1e-6) - 1.)
     
        return torch.mean(dimwise_kld)
     
  
    
    def CrossEnt_loss(self, logits, y_true):
        loss = torch.mean(-y_true*torch.log(logits + 1e-6))
        return loss
    
    def clf_CrossEnt_loss(self, logits, y_true):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss
   

    def encode_step(self, x,c):
        mu, logvar  = self.Encoder(x,c)
        return mu, logvar

    def decode_step(self, enc, c,label):
        mu, logvar = self.Decoder[label](enc, c)
        return mu, logvar

    def generator_step(self, enc,c,label):
        gen_mu, gen_logvar = self.Generators(enc,c)
        return gen_mu,gen_logvar

    def entropy_loss(self, logits):
        loss = torch.mean(-logits*logits.log())
        return loss

    def clf_step(self, input,label,batch_size):
        device = torch.device("cuda")
        src_c = self.label_generate(label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        en_mu,en_lv = self.encode_step(input,src_c)
        z = self.reparameterize(en_mu,en_lv)
        logits = self.SpeakerClassifier(z)
        c = self.label_generate(label,batch_size).type(torch.LongTensor).to(self.device)
        loss = self.clf_CrossEnt_loss(logits,c)
        return loss

    def asr_step(self, input,label,batch_size):
        device = torch.device("cuda")
        src_c = self.label_generate(label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        en_mu,en_lv = self.encode_step(input,src_c)
        z = self.reparameterize(en_mu,en_lv)
        logits = self.ASRLayer(z)
        c = self.label_generate(label,batch_size).type(torch.LongTensor).to(self.device)
        loss = self.entropy_loss(logits)
        return loss

    def patch_step(self, x, x_tilde,trg_num, is_dis=True):
        device = torch.device("cuda")
        trg_c = self.label_generate(trg_num,8)
        trg_c = self.label2onehot(trg_c,8).to(device, dtype=torch.float)
        D_real = self.Discriminator(x, trg_c,classify=False)
        D_fake = self.Discriminator(x_tilde, trg_c,classify=False)
        
        if is_dis:
            w_dis = -torch.mean(D_real) + torch.mean(D_fake)  
            return w_dis
        else:
            return - torch.mean(D_fake)

    def label_generate(self,num,batch):
        label = []
        for i in range(batch):
            label.append(num)
        return torch.Tensor(label)

    def label2onehot(self, labels, batch_size):
       
        shape = labels.shape
        labels = torch.Tensor(labels)
        out = torch.zeros(batch_size, 4)
        for i in range(shape[0]):
            out[i, int(labels[i])] = 1
        return out

    def clf_asr_step(self, input,target,src_label,trg_label,batch_size):
        device = torch.device("cuda")
        s_label_list = self.label_generate(src_label,batch_size).type(torch.LongTensor).to(device)
        src_c = self.label_generate(src_label,batch_size)
        trg_c = self.label_generate(trg_label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
        en_mu,en_lv = self.encode_step(input,src_c)
        KLD = self.KLD_loss(en_mu,en_lv)
        z = self.reparameterize(en_mu,en_lv)
        asr_logits = self.ASRLayer(z)
        asr_loss = self.entropy_loss(asr_logits)
        spk_logits = self.SpeakerClassifier(z)
        clf_loss = self.clf_CrossEnt_loss(spk_logits,s_label_list)
        same_xt_mu,same_xt_lv = self.decode_step(z, src_c,src_label)
        same_x_tilde = self.reparameterize(same_xt_mu,same_xt_lv)
        same_loss_rec = self.GaussianLogDensity(input,same_xt_mu,same_xt_lv)
        return clf_loss, asr_loss

    def vae_step(self, input,target,src_label,trg_label,batch_size):
        device = torch.device("cuda")
        src_c = self.label_generate(src_label,batch_size)
        trg_c = self.label_generate(trg_label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
    
        en_mu,en_lv = self.encode_step(input,src_c)
        z = self.reparameterize(en_mu,en_lv)
        xt_mu,xt_lv = self.decode_step(z, trg_c,trg_label)
        x_tilde = self.reparameterize(xt_mu,xt_lv)

        ###loss
        KLD = self.KLD_loss(en_mu,en_lv)
        loss_rec = self.GaussianLogDensity(input,xt_mu,xt_lv)
        return KLD,-loss_rec,x_tilde


    def cycle_step(self, input,target,src_label,trg_label,batch_size):
        device = torch.device("cuda")
    
        src_c = self.label_generate(src_label,batch_size)
        trg_c = self.label_generate(trg_label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
  
        en_mu,en_lv = self.encode_step(input,src_c)
        z = self.reparameterize(en_mu,en_lv)
        
        
        convert_xt_mu,convert_xt_lv = self.decode_step(z, trg_c,trg_label)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
        ###cycle step
        cyc_en_mu,cyc_en_lv = self.encode_step(convert_x_tilde,trg_c)
        cyc_z = self.reparameterize(cyc_en_mu,cyc_en_lv)
        cyc_xt_mu,cyc_xt_lv = self.decode_step(cyc_z, src_c,src_label)
        cyc_x_tilde = self.reparameterize(cyc_xt_mu,cyc_xt_lv)
        
        
        
        cyc_loss_rec = self.GaussianLogDensity(input,cyc_xt_mu,cyc_xt_lv)
    
        cyc_KLD = self.KLD_loss(cyc_en_mu,cyc_en_lv)
        KLD_same_check = torch.mean(torch.abs(en_mu - cyc_en_mu))

        return cyc_KLD, -cyc_loss_rec 

    def sem_step(self, input,target,src_label,trg_label,batch_size):
        device = torch.device("cuda")

        src_c = self.label_generate(src_label,batch_size)
        trg_c = self.label_generate(trg_label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
      
        en_mu,en_lv = self.encode_step(input,src_c)
        z = self.reparameterize(en_mu,en_lv)
    
        
        
        
        convert_xt_mu,convert_xt_lv = self.decode_step(z, trg_c,trg_label)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
        ###cycle step
        cyc_en_mu,cyc_en_lv = self.encode_step(convert_x_tilde,trg_c)
        cyc_z = self.reparameterize(cyc_en_mu,cyc_en_lv)
      
        KLD_same_check = torch.mean(torch.abs(z - cyc_z).pow(2))
       
        return KLD_same_check 

    def GaussianLogDensity(self,x, mu, log_var):
            c = torch.log(2.*torch.from_numpy(np.array(3.141592)))
            var = torch.exp(log_var)
            x_mu2 = (x - mu).pow(2)   
            x_mu2_over_var = torch.div(x_mu2, var + 1e-6)
            log_prob = -0.5 * (c + log_var + x_mu2_over_var)
            return torch.mean(log_prob)

    def AC_step(self, input,target,src_label,trg_label,batch_size):
        device = torch.device("cuda")
      
        src_c = self.label_generate(src_label,batch_size)
        trg_c = self.label_generate(trg_label,batch_size)
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)


        s_label_list = self.label_generate(src_label,batch_size).type(torch.LongTensor).to(device)
        t_label_list = self.label_generate(trg_label,batch_size).type(torch.LongTensor).to(device)
      
        acc_s,src_t_label = self.ACLayer(input)
        acc_t,trg_t_label = self.ACLayer(target)
        
        AC_source =  self.CrossEnt_loss(src_t_label, src_c)
        AC_target =  self.CrossEnt_loss(trg_t_label, trg_c)
        return AC_source,AC_target
        
    def AC_F_step(self, input,target,src_label,trg_label,batch_size):
       
        device = torch.device("cuda")
        
        src_c = self.label_generate(src_label,batch_size)
    
        src_c = self.label2onehot(src_c,batch_size).to(device, dtype=torch.float)
      
        en_mu,en_lv = self.encode_step(input,src_c)
     
        z = self.reparameterize(en_mu,en_lv)
                
        acc_s,t_label = self.ACLayer(input)
       
        AC_real =  self.CrossEnt_loss(t_label, src_c)
      
        trg_c = self.label_generate(0,batch_size)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
        convert_xt_mu,convert_xt_lv = self.decode_step(z, trg_c,0)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
        acc_conv_t,c_label = self.ACLayer(convert_x_tilde)
        AC_cross_1 = self.CrossEnt_loss(c_label, trg_c)
        
        trg_c = self.label_generate(1,batch_size)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
        convert_xt_mu,convert_xt_lv = self.decode_step(z, trg_c,1)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
        acc_conv_t,c_label = self.ACLayer(convert_x_tilde)
        AC_cross_2 = self.CrossEnt_loss(c_label, trg_c)

        trg_c = self.label_generate(2,batch_size)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
        convert_xt_mu,convert_xt_lv = self.decode_step(z, trg_c,2)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
        acc_conv_t,c_label = self.ACLayer(convert_x_tilde)
        AC_cross_3 = self.CrossEnt_loss(c_label, trg_c)

        trg_c = self.label_generate(3,batch_size)
        trg_c = self.label2onehot(trg_c,batch_size).to(device, dtype=torch.float)
        convert_xt_mu,convert_xt_lv = self.decode_step(z, trg_c,3)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
        acc_conv_t,c_label = self.ACLayer(convert_x_tilde)
        AC_cross_4 = self.CrossEnt_loss(c_label, trg_c)
     
        AC_cross = (AC_cross_1+AC_cross_2+AC_cross_3+AC_cross_4)/4.0
        return AC_real,AC_cross

  

    def train(self,batch_size, mode='train',model_iter='0'):
        speaker_list = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2']
        num_mcep = 36
        frame_period = 5.0
        n_frames = 128
        batch_num = batch_size
      

        exp_dir = os.path.join('processed')
        device = torch.device("cuda")
       
      
        print('Loading cached data...')
        
        ac_lr = 0.0001
        lr = 0.001
        random.seed()
       
        
        if mode == 'DisentanglementANDConvPath_VAE':
            for ep in range(100):
                
                for i in range(4):
                    for j in range(4):
                    
                        src_speaker = speaker_list[i]
                      
                        trg_speaker = speaker_list[j]
                        
                        exp_A_dir = os.path.join(exp_dir, src_speaker)
                        exp_B_dir = os.path.join(exp_dir, trg_speaker)
                        
                        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
                            os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
                        coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
                            os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))
                        dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)

                        dataset_A = np.expand_dims(dataset_A, axis=1)
                        dataset_A = torch.from_numpy(dataset_A).to(device, dtype=torch.float)
                        dataset_B = np.expand_dims(dataset_B, axis=1)
                        dataset_B = torch.from_numpy(dataset_B).to(device, dtype=torch.float)
                        for iteration in range(4):
                            start = iteration * batch_num
                            end = (iteration + 1) * batch_num
                            if ((iteration+1) % 4)!=0 :
                                self.grad_reset()
                                clf_loss_A = self.clf_step(dataset_A[start:end], i, batch_num)
                                clf_loss_B = self.clf_step(dataset_B[start:end], j, batch_num)
                                Clf_loss = clf_loss_A + clf_loss_B
                                loss = Clf_loss
                                loss.backward()
                                self.cls_optimizer.step()
                               
                            elif ((iteration+1) % 4)==0 :
                                self.grad_reset()
                                asr_loss_A = self.asr_step(dataset_A[start:end], i, batch_num)
                                asr_loss_B = self.asr_step(dataset_B[start:end], j, batch_num)
                                asr_loss = asr_loss_A + asr_loss_B
                                loss = asr_loss
                                
                                loss.backward()
                                self.asr_optimizer.step()

                                self.grad_reset()
                                AC_source,AC_target = \
                                        self.AC_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                AC_t_loss = AC_source+AC_target
                                
                                AC_t_loss.backward()
                               
                                self.ac_optimizer.step()
                               
                                self.grad_reset()

                                ###VAE step
                                src_KLD, src_same_loss_rec, _ = self.vae_step(dataset_A[start:end],dataset_B[start:end],i,i,batch_num)
                                trg_KLD, trg_same_loss_rec, _ = self.vae_step(dataset_B[start:end],dataset_A[start:end],j,j,batch_num)


                                ###AC F step
                                AC_real_src,AC_cross_src = \
                                        self.AC_F_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                AC_real_trg,AC_cross_trg = \
                                        self.AC_F_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)


                                ###clf asr step
                                clf_loss_A,asr_loss_A = \
                                    self.clf_asr_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                clf_loss_B,asr_loss_B = \
                                    self.clf_asr_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)
                                Clf_loss = (clf_loss_A + clf_loss_B)/2.0
                                
                                ASR_loss = (asr_loss_A + asr_loss_B)/2.0

                                ###Cycle step
                                src_cyc_KLD, src_cyc_loss_rec = self.cycle_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                trg_cyc_KLD, trg_cyc_loss_rec = self.cycle_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)


                                ###Semantic step
                                src_semloss = self.sem_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                trg_semloss = self.sem_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)


                                
                                AC_f_loss = (AC_real_src+AC_real_trg+AC_cross_src+AC_cross_trg)/4.0
                                Sem_loss = (src_semloss+trg_semloss)/2.0
                                Cycle_KLD_loss = (src_cyc_KLD + trg_cyc_KLD)/2.0
                                Cycle_rec_loss = (src_cyc_loss_rec + trg_cyc_loss_rec)/2.0
                                KLD_loss = (src_KLD+trg_KLD)/2.0                         
                                Rec_loss = (src_same_loss_rec+trg_same_loss_rec)/2.0
                                loss = Rec_loss + KLD_loss+Cycle_KLD_loss+Cycle_rec_loss + AC_f_loss+Sem_loss-Clf_loss+ASR_loss
                                loss.backward()
                                self.vae_optimizer.step()
                
                if (ep+1)%1==0:
                    print("Epoch : {}, Recon : {:.3f}, KLD : {:.3f}, AC t Loss : {:.3f}, AC f Loss : {:.3f}, Sem Loss : {:.3f}, Clf : {:.3f}, Asr Loss : {:.3f}"\
                        .format(ep+1,Rec_loss,KLD_loss,AC_t_loss,AC_cross_trg,Sem_loss,Clf_loss,ASR_loss))
                os.makedirs("./VAE_all"+model_iter, exist_ok=True)
                if (ep+1) % 50 ==0:
                        print("Model Save Epoch {}".format(ep+1))
                        self.save_model("VAE_all"+model_iter, ep+1)
                
        if mode == 'DisentanglementANDConvPath_VAE_with_GAN':
            os.makedirs("./GAN_all"+model_iter, exist_ok=True)
            for ep in range(200):
                
                if ep>100:
                    
                    lr = lr * 0.9
               
                for i in range(4):
                    for j in range(4):
                     
                        src_speaker = speaker_list[i]
                        
                        trg_speaker = speaker_list[j]
                        
                        exp_A_dir = os.path.join(exp_dir, src_speaker)
                        exp_B_dir = os.path.join(exp_dir, trg_speaker)
                        
                        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
                            os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
                        coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
                            os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))
                        dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)

                        dataset_A = np.expand_dims(dataset_A, axis=1)
                        dataset_A = torch.from_numpy(dataset_A).to(device, dtype=torch.float)
                        dataset_B = np.expand_dims(dataset_B, axis=1)
                        dataset_B = torch.from_numpy(dataset_B).to(device, dtype=torch.float)
                        for iteration in range(81//batch_num):
                            start = iteration * batch_num
                            end = (iteration+1) * batch_num

                            if ((iteration+1)%5)!=0:
                               
                                self.grad_reset()
                                clf_loss_A = self.clf_step(dataset_A[start:end], i, batch_num)
                                clf_loss_B = self.clf_step(dataset_B[start:end], j, batch_num)
                                Clf_loss = clf_loss_A + clf_loss_B
                                loss = Clf_loss
                               
                                
                                loss.backward()
                                
                                
                                self.cls_optimizer.step()


                                self.grad_reset()
                                convert_KLD, convert_rec,src_to_trg_x_tilde = self.vae_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                
                                trg_w_dis = self.patch_step(dataset_B[start:end], src_to_trg_x_tilde,j, is_dis=True)
                                
                               
                                trg_adv_loss = trg_w_dis
                                adv_loss =  (trg_adv_loss)

                                
                                adv_loss.backward()
                               
                                self.dis_optimizer.step()
                                for p in self.Discriminator.parameters():
                                        p.data.clamp_(-0.01, 0.01)

                            
                            elif ((iteration+1)%5)==0 and ep>10:
                                self.grad_reset()
                                asr_loss_A = self.asr_step(dataset_A[start:end], i, batch_num)
                                asr_loss_B = self.asr_step(dataset_B[start:end], j, batch_num)
                                asr_loss = asr_loss_A + asr_loss_B
                                loss = asr_loss
                               
                                loss.backward()
                                self.asr_optimizer.step()

                                self.grad_reset()
                                AC_source,AC_target = \
                                        self.AC_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                AC_t_loss = AC_source+AC_target
                                
                                AC_t_loss.backward()
                              
                                self.ac_optimizer.step()
                                
                                self.grad_reset()

                                ###VAE step
                                src_KLD, src_same_loss_rec, _ = self.vae_step(dataset_A[start:end],dataset_B[start:end],i,i,batch_num)
                                trg_KLD, trg_same_loss_rec, _ = self.vae_step(dataset_B[start:end],dataset_A[start:end],j,j,batch_num)


                                ###AC F step
                                AC_real_src,AC_cross_src = \
                                        self.AC_F_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                AC_real_trg,AC_cross_trg = \
                                        self.AC_F_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)


                                ###clf asr step
                                clf_loss_A,asr_loss_A = \
                                    self.clf_asr_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                clf_loss_B,asr_loss_B = \
                                    self.clf_asr_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)
                                Clf_loss = (clf_loss_A + clf_loss_B)/2.0
                                
                                ASR_loss = (asr_loss_A + asr_loss_B)/2.0

                                ###Cycle step
                                src_cyc_KLD, src_cyc_loss_rec = self.cycle_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                trg_cyc_KLD, trg_cyc_loss_rec = self.cycle_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)


                                ###Semantic step
                                src_semloss = self.sem_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                                trg_semloss = self.sem_step(dataset_B[start:end],dataset_A[start:end],j,i,batch_num)

                                convert_KLD, convert_rec,src_to_trg_x_tilde = self.vae_step(dataset_A[start:end],dataset_B[start:end],i,j,batch_num)
                              
                                trg_loss_adv = self.patch_step(dataset_B[start:end], src_to_trg_x_tilde,j, is_dis=False)

                               
                                AC_f_loss = (AC_real_src+AC_real_trg+AC_cross_src+AC_cross_trg)/4.0
                               
                                Sem_loss = (src_semloss+trg_semloss)/2.0
                                Cycle_KLD_loss = (src_cyc_KLD + trg_cyc_KLD)/2.0
                                Cycle_rec_loss = (src_cyc_loss_rec + trg_cyc_loss_rec)/2.0
                                KLD_loss = (src_KLD+trg_KLD)/2.0
                                                          
                                Rec_loss = (src_same_loss_rec+trg_same_loss_rec)/2.0
                               
                                
                                loss = 2*(Rec_loss + KLD_loss)+Cycle_rec_loss+Cycle_KLD_loss + AC_f_loss+Sem_loss+trg_loss_adv -Clf_loss+ASR_loss
                              
                                loss.backward()
                              
                                self.vae_optimizer.step()
                               
                if ep>10:
                    print("Epoch : {}, Recon Loss : {:.3f},  KLD Loss : {:.3f}, Dis Loss : {:.3f},  GEN Loss : {:.3f}, AC t Loss : {:.3f}, AC f Loss : {:.3f}".format(ep,Rec_loss,KLD_loss,adv_loss,trg_loss_adv,AC_t_loss,AC_cross_trg))   
                else:
                    print("Epoch : {} Dis Loss : {}".format(ep,adv_loss))   
             
                if (ep+1) % 50 ==0:   
                    print("Model Save Epoch {}".format(ep+1))
                    self.save_model("GAN_all"+model_iter, ep+1)
                
                
