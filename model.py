from turtle import forward
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

class LSTM(nn.Module):
    def __init__(self,  hidden_size = 256, num_action = 4, n_feature = 512):
        super(LSTM, self).__init__()

        self.img_encoder = nn.Linear(n_feature, hidden_size)
        self.action_encoder = nn.Linear(num_action, hidden_size)
        self.working_memory = nn.LSTM(hidden_size , hidden_size)
        self.actor = nn.Linear(hidden_size, num_action)


    def forward(self, feature, action,  mem_state):
        feature = self.img_encoder(feature)
        action = self.action_encoder(action)
        mem_input = feature + action

        h_t, mem_state = self.working_memory(mem_input, mem_state)
        action_logits = self.actor(h_t)

        return action_logits,  mem_state

    def get_init_states(self, device='cpu'):
        h0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        c0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        return (h0, c0) 

class LSTM_Attn(nn.Module):


    def __init__(self, channel_size = 64, att_size = 128,  hidden_size = 256, num_actions = 4):
        super(LSTM_Attn, self).__init__()
        self.att_size = att_size
        self.channel_size = channel_size

        self.img_encoder = nn.Conv2d(channel_size, att_size, 1)
        self.action_encoder = nn.Linear(num_actions, hidden_size)

        self.img_att = nn.Linear(att_size, att_size)
        self.h_att = nn.Linear(hidden_size, att_size)
        self.full_att = nn.Linear(att_size, 1)
        self.softmax = nn.Softmax(dim=1)

        self.working_memory = nn.LSTM(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size + att_size, num_actions)

    def forward(self, img_code, action, mem_state):
        action_input = self.action_encoder(action) 
        mem_input = action_input 

        h_t, mem_state = self.working_memory(mem_input, mem_state)

        img_code = F.relu(self.img_encoder(img_code))
        img_code = img_code.permute(0, 2, 3, 1).view(1, -1, self.att_size)
        att1 = F.relu(self.img_att(img_code))
        att2 = F.relu(self.h_att(h_t))
        att = self.full_att(att1 + att2).squeeze(2)
        alpha = self.softmax(att)
        att_out = (img_code * alpha.unsqueeze(2)).sum(dim=1).unsqueeze(0) # (batch_size, 1, att_size)
        

        action_logits = self.actor(T.cat((att_out, h_t), dim=2)) # (batch_size, 1, num_action)
        # value_estimate = self.critic(T.cat((att_out, h_t), dim=2))

        return action_logits, mem_state

    def get_init_states(self, device='cpu'):
        h0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        c0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        return (h0, c0) 


