import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.label_num = 4
        self.conv1 = nn.Conv2d(1 , 5, (3,9), (1,1), padding=(1, 4))
        self.conv1_bn = nn.BatchNorm2d(5)
        self.conv1_gated = nn.Conv2d(1, 5, (3,9), (1,1), padding=(1,4))
        self.conv1_gated_bn = nn.BatchNorm2d(5)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(5 , 10, (4,8), (2,2), padding=(1, 3))
        self.conv2_bn = nn.BatchNorm2d(10)
        self.conv2_gated = nn.Conv2d(5 , 10, (4,8), (2,2), padding=(1, 3))
        self.conv2_gated_bn = nn.BatchNorm2d(10)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(10, 10, (4,8), (2,2), padding=(1, 3))
        self.conv3_bn = nn.BatchNorm2d(10)
        self.conv3_gated = nn.Conv2d(10 , 10, (4,8), (2,2), padding=(1, 3))
        self.conv3_gated_bn = nn.BatchNorm2d(10)
        self.conv3_sigmoid = nn.Sigmoid()
    
        self.conv4 = nn.Conv2d(10, 16, (9,5), (9,1), padding=(0, 2))
      
    def id_bias_add_2d(self, inputs, id):
        id = id.view(id.size(0), id.size(1), 1, 1)
        id = id.repeat(1, 1, inputs.size(2), inputs.size(3))
        inputs_bias_added = torch.cat([inputs, id], dim=1)
        return inputs_bias_added
     
    def forward(self, x,label):
        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 

        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
       
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
       
        h4 = self.conv4(h3)
        h4_mu = h4[:,:8,:,:]
       
        h4_logvar = h4[:,8:,:,:]
       
        return h4_mu, h4_logvar 


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.label_num = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        self.upconv1 = nn.ConvTranspose2d(8 , 10, (9,5), (9,1), padding=(0, 2))
        self.upconv1_bn = nn.BatchNorm2d(10)
        self.upconv1_gated = nn.ConvTranspose2d(8, 10, (9,5), (9,1), padding=(0, 2))
        self.upconv1_gated_bn = nn.BatchNorm2d(10)
        self.upconv1_sigmoid = nn.Sigmoid()
      
        self.upconv2 = nn.ConvTranspose2d(10 ,10, (4,8), (2,2), padding=(1, 3))
        self.upconv2_bn = nn.BatchNorm2d(10)
        self.upconv2_gated = nn.ConvTranspose2d(10 , 10, (4,8), (2,2), padding=(1, 3))
        self.upconv2_gated_bn = nn.BatchNorm2d(10)
        self.upconv2_sigmoid = nn.Sigmoid()
       
        self.upconv3 = nn.ConvTranspose2d(10, 5, (4,8), (2,2), padding=(1, 3))
        self.upconv3_bn = nn.BatchNorm2d(5)
        self.upconv3_gated = nn.ConvTranspose2d(10 , 5, (4,8), (2,2), padding=(1, 3))
        self.upconv3_gated_bn = nn.BatchNorm2d(5)
        self.upconv3_sigmoid = nn.Sigmoid()
       
        self.upconv4 = nn.ConvTranspose2d(5 , 2, (3,9), (1,1), padding=(1, 4))
    
    def id_bias_add_2d(self, inputs, id):
        id = id.view(id.size(0), id.size(1), 1, 1)
        id = id.repeat(1, 1, inputs.size(2), inputs.size(3))
        inputs_bias_added = torch.cat([inputs, id], dim=1)
        return inputs_bias_added
 
    def forward(self, z,label):
        h5_ = self.upconv1_bn(self.upconv1(z))
        h5_gated = self.upconv1_gated_bn(self.upconv1(z))
        h5 = torch.mul(h5_, self.upconv1_sigmoid(h5_gated)) 
       
        h6_ = self.upconv2_bn(self.upconv2(h5))
        h6_gated = self.upconv2_gated_bn(self.upconv2(h5))
        h6 = torch.mul(h6_, self.upconv2_sigmoid(h6_gated)) 
       
        h7_ = self.upconv3_bn(self.upconv3(h6))
        h7_gated = self.upconv3_gated_bn(self.upconv3(h6))
        h7 = torch.mul(h7_, self.upconv3_sigmoid(h7_gated)) 
       
        h8 = self.upconv4(h7)
        h8_mu = h8[:,:1,:,:]
      
        h8_logvar = h8[:,1:,:,:]
        return h8_mu, h8_logvar




class SpeakerClassifier(nn.Module):
    def __init__(self):
        super(SpeakerClassifier, self).__init__()
        self.label_num = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(8, 16, (1,4), (1,2), padding=(0, 1))
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv1_gated = nn.Conv2d(8, 16, (1,4), (1,2), padding=(0, 1))
        self.conv1_gated_bn = nn.BatchNorm2d(16)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(16, 32, (1,4), (1,2), padding=(0, 1))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_gated = nn.Conv2d(16, 32, (1,4), (1,2), padding=(0, 1))
        self.conv2_gated_bn = nn.BatchNorm2d(32)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(32, 32, (1,4), (1,2), padding=(0, 1))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_gated = nn.Conv2d(32, 32, (1,4), (1,2), padding=(0,1))
        self.conv3_gated_bn = nn.BatchNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()

        self.conv4 = nn.Conv2d(32, 16, (1,4), (1,2), padding=(0, 1))
        self.conv4_bn = nn.BatchNorm2d(16)
        self.conv4_gated = nn.Conv2d(32, 16, (1,4), (1,2), padding=(0, 1))
        self.conv4_gated_bn = nn.BatchNorm2d(16)
        self.conv4_sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(16, 1, (1, 5),(1,1), padding=(0, 1))
        self.conv_out_sigmoid = nn.Sigmoid()
        self.conv_classify = nn.Conv2d(16, self.label_num, (1, 3),(1,1), padding=(0, 1))
        self.linear = nn.Linear(8, self.label_num)
        

    def forward(self, input,classify=True):
        h1_ = self.conv1_bn(self.conv1(input))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(input))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
    
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
    
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
    
        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated))
    
        if classify:
            logits = self.conv_classify(h4)
            logits = logits.view(logits.size(0), -1)
            logits = self.linear(logits)
            return  logits
        else:
            return mean_val
      
class ASRLayer(nn.Module):
    def __init__(self):
        super(ASRLayer, self).__init__()
        self.label_num = 4
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(8, 16, (1,4), (1,1), padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv1_gated = nn.Conv2d(8, 16, (1,4), (1,1), padding=(1, 1))
        self.conv1_gated_bn = nn.BatchNorm2d(16)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(16, 32, (1,4), (1,2), padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_gated = nn.Conv2d(16, 32, (1,4), (1,2), padding=(1, 1))
        self.conv2_gated_bn = nn.BatchNorm2d(32)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(32, 32, (1,4), (1,2), padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_gated = nn.Conv2d(32, 32, (1,4), (1,2), padding=(1,1))
        self.conv3_gated_bn = nn.BatchNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()

        self.conv4 = nn.Conv2d(32, 16, (1,4), (1,2), padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(16)
        self.conv4_gated = nn.Conv2d(32, 16, (1,4), (1,2), padding=(1, 1))
        self.conv4_gated_bn = nn.BatchNorm2d(16)
        self.conv4_sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(16, 1, (1, 5),(1,1), padding=(0, 1))
        self.conv_out_sigmoid = nn.Sigmoid()
        self.conv_classify = nn.Conv2d(16, 1, (1, 5),(1,1), padding=(0, 1))
        self.linear = nn.Linear(9, 1)
        
    def forward(self, input,classify=True):
        h1_ = self.conv1_bn(self.conv1(input))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(input))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 

        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated)) 
        
        val = self.conv_out_sigmoid(self.conv_out(h4))
        val = val.view(val.size(0), -1)
       
        if classify:
           
            logits = self.conv_classify(h4)
            logits = logits.view(logits.size(0), -1)
            logits = self.linear(logits)
            logits = self.conv_out_sigmoid(logits)
         
            return  logits
        else:
            return mean_val
      

class ACLayer(nn.Module):
    def __init__(self):
        super(ACLayer, self).__init__()
        self.label_num = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 8, (4,4), (2,2), padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(1, 8, (4,4), (2,2), padding=(1,1))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8, 16, (4,4), (2,2), padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8, 16, (4,4), (2,2), padding=(1, 1))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16, 32, (4,4), (2,2), padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_gated = nn.Conv2d(16, 32, (4,4), (2,2), padding=(1, 1))
        self.conv3_gated_bn = nn.BatchNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()

        self.conv4 = nn.Conv2d(32, 16, (3,4), (1,2), padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(16)
        self.conv4_gated = nn.Conv2d(32, 16, (3,4), (1,2), padding=(1, 1))
        self.conv4_gated_bn = nn.BatchNorm2d(16)
        self.conv4_sigmoid = nn.Sigmoid()
      
        self.conv_classify = nn.Conv2d(16, self.label_num, (1, 4),(1,2), padding=(0, 1),bias=False)
        self.conv_softmax = nn.Softmax(dim=1)
      

    def forward(self, input,classify=True):
        input = input[:,:,:8,:]
        
        h1_ = self.conv1_bn(self.conv1(input))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(input))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
      
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
       
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
      
        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated)) 
     
        if classify:
           
            h5 = self.conv_classify(h4)
           
            h5 = self.conv_softmax(h5)
       
            prod_logit = torch.prod(h5, dim=-1, keepdim=False)
        
            logits = prod_logit.view(prod_logit.size()[0], self.label_num)
         
            return prod_logit,logits
        else:
            return mean_val


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_num = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1+self.label_num, 8, (4,4), (2,2), padding=(1, 1))
        self.conv1_bn = nn.InstanceNorm2d(8)
        self.conv1_gated = nn.Conv2d(1+self.label_num, 8, (4,4), (2,2), padding=(1, 1))
        self.conv1_gated_bn = nn.InstanceNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8+self.label_num, 16, (4,4), (2,2), padding=(1, 1))
        self.conv2_bn = nn.InstanceNorm2d(16)
        self.conv2_gated = nn.Conv2d(8+self.label_num, 16, (4,4), (2,2), padding=(1, 1))
        self.conv2_gated_bn = nn.InstanceNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16+self.label_num, 32, (4,4), (2,2), padding=(1, 1))
        self.conv3_bn = nn.InstanceNorm2d(32)
        self.conv3_gated = nn.Conv2d(16+self.label_num, 32, (4,4), (2,2), padding=(1, 1))
        self.conv3_gated_bn = nn.InstanceNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()

        self.conv4 = nn.Conv2d(32+self.label_num, 16, (3,4), (1,2), padding=(1, 1))
        self.conv4_bn = nn.InstanceNorm2d(16)
        self.conv4_gated = nn.Conv2d(32+self.label_num, 16, (3,4), (1,2), padding=(1, 1))
        self.conv4_gated_bn = nn.InstanceNorm2d(16)
        self.conv4_sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(16+self.label_num, 1, (1, 4),(1,2), padding=(0, 1))
     
        self.conv_classify = nn.Conv2d(16+self.label_num, self.label_num, (36, 5),(36,1))
      
        self.linear = nn.Linear(16, 1)
      
    def id_bias_add_2d(self, inputs, id):
       
        id = id.view(id.size(0), id.size(1), 1, 1)
        id = id.repeat(1, 1, inputs.size(2), inputs.size(3))
        inputs_bias_added = torch.cat([inputs, id], dim=1)
        return inputs_bias_added
    
    def forward(self, input,label,classify=False):
        input = self.id_bias_add_2d(input,label)
        h1_ = self.conv1_bn(self.conv1(input))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(input))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        h1 = self.id_bias_add_2d(h1,label)
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        h2 = self.id_bias_add_2d(h2,label)
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
        h3 = self.id_bias_add_2d(h3,label)
        
        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated)) 
        h4 = self.id_bias_add_2d(h4,label)
       
        val = self.conv_out(h4)
        val = val.view(val.size(0), -1)
        mean_val = self.linear(val)
        
        if classify:
          
            logits = self.conv_classify(h4)
            logits = logits.view(logits.size(0), -1)
            return mean_val, logits
        else:
            return mean_val
       