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
        

class A2C_LSTM2(nn.Module):
    def __init__(self, channel_size = 9, code_size = 128, hidden_size = 256, num_actions = 4):
        super(A2C_LSTM2, self).__init__()

        # LSTM input: timestamp + one-hot action code + previous reward
        self.conv = nn.Conv2d(1, channel_size, 2)
        self.img_encoder = nn.Linear(channel_size * 8 * 8, code_size)
        self.action_encoder = nn.Linear(num_actions, code_size)
        self.working_memory = nn.LSTM(code_size, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, pre_input, img = True, mem_state = None):
        pre_input = pre_input.unsqueeze(0).unsqueeze(0)
        if img:
            code = F.relu(self.conv(pre_input)).flatten()
            code = self.img_encoder(code).unsqueeze(0).unsqueeze(0)
        else:
            code = self.action_encoder(pre_input)

        if mem_state is None:
            mem_state = self.get_init_states()  
        
        mem_input = code

        h_t, mem_state = self.working_memory(mem_input, mem_state)

        action_logits = self.actor(h_t)
        value_estimate = self.critic(h_t)

        return action_logits, value_estimate, mem_state

    def get_init_states(self, device='cpu'):
        h0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        c0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        return (h0, c0) 


class A2C_LSTM3(nn.Module):
    def __init__(self, channel_size = 100, code_size = 256, hidden_size = 256, num_actions = 4):
        super(A2C_LSTM3, self).__init__()

        # LSTM input: timestamp + one-hot action code + previous reward
        self.conv1 = nn.Conv2d(1, channel_size, 2)
        self.conv2 = nn.Conv2d(channel_size, channel_size, 2)
        self.conv3 = nn.Conv2d(channel_size, channel_size, 2)
        self.img_encoder = nn.Linear(channel_size * 8 * 8, code_size)
        self.action_encoder = nn.Linear(num_actions, code_size)
        self.working_memory = nn.LSTM(code_size, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        # self.estimator = nn.Linear(hidden_size, 9*9)
        # self.filter = nn.Linear(hidden_size, code_size*9*9)
        self.code_size = code_size

        # intialize actor and critic weights
        T.nn.init.orthogonal_(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        T.nn.init.orthogonal_(self.critic.weight.data, 1)
        self.critic.bias.data.fill_(0)

    def forward(self, pre_input, img = False, act = False, mem_state = None):
        pre_input = pre_input.unsqueeze(0).unsqueeze(0)
        if img:
            # code = F.relu(self.conv1(pre_input))
            # code = F.relu(self.conv2(code))
            code = F.relu(self.conv1(pre_input)).flatten()
            code = self.img_encoder(code).unsqueeze(0).unsqueeze(0)
        elif act:
            code = self.action_encoder(pre_input)
        else:
            code = pre_input

        if mem_state is None:
            mem_state = self.get_init_states()  
        
        mem_input = code

        h_t, mem_state = self.working_memory(mem_input, mem_state)

        action_logits = self.actor(h_t)
        value_estimate = self.critic(h_t)
        value_mat = F.relu(self.estimator(h_t)).flatten()
        iter_mat = F.relu(self.filter(h_t)).reshape(self.code_size, 9*9)

        return action_logits, value_estimate, mem_state, value_mat, iter_mat

    def get_init_states(self, device='cpu'):
        h0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        c0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        return (h0, c0) 




class EncoderRNN(nn.Module):
    def __init__(self, n_state = 49, n_out=49, code_size = 128, hidden_size = 256, num_actions = 4, dropout_p=.1):
        super(EncoderRNN, self).__init__()

        self.code_size = code_size
        self.state_encoder = nn.Linear(n_state, code_size)
        self.action_encoder = nn.Linear(num_actions, code_size)
        self.working_memory = nn.LSTM(code_size * 2, hidden_size)
        self.state_decoder = nn.Linear(hidden_size, n_out)
        self.dropout = nn.Dropout(dropout_p)
        

    def forward(self, pre_state, action,  mem_state = None):
        state_code = self.state_encoder(pre_state.unsqueeze(0).unsqueeze(0))
        action_code = self.action_encoder(action.unsqueeze(0).unsqueeze(0))
        mem_input = T.cat((state_code, action_code), dim=-1) # (1, 1, code_size*2)
        mem_input = self.dropout(mem_input)
        if mem_state is None:
            mem_state = self.get_init_states()  

        h_t, mem_state = self.working_memory(mem_input, mem_state)
        state_logit = self.state_decoder(h_t)

        return h_t, mem_state, state_logit

    def get_init_states(self, device='cpu'):
        h0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        c0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        return (h0, c0) 


class AttnEncoderRNN(nn.Module):
    def __init__(self, length, n_state, n_action=4, code_size=128, hidden_size=256, dropout_p=.1):
        super(AttnEncoderRNN, self).__init__()
        self.state_encoder = nn.Linear(n_state, code_size)
        self.action_encoder = nn.Linear(n_action, code_size)
        self.dropout = nn.Dropout(dropout_p)
        self.working_memory = nn.LSTM(code_size*2, hidden_size)
        self.state_decoder = nn.Linear(hidden_size, n_state)
        self.attn = nn.Linear(code_size*2 + hidden_size, length)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)

    # encoder_output: (length, hidden_size)
    def forward(self, pre_state, action, mem_state, encoder_output):
        state_code = self.state_encoder(pre_state.unsqueeze(0).unsqueeze(0))
        action_code = self.action_encoder(action.unsqueeze(0).unsqueeze(0))
        mem_input = T.cat((state_code, action_code), dim=-1)
        mem_input = self.dropout(mem_input)

        attn_weight = self.attn(T.cat((mem_input[0], mem_state[0][0]), dim=1))
        attn_weight = F.softmax(attn_weight, dim=1).unsqueeze(0)
        attn_applied = T.bmm(attn_weight, encoder_output.unsqueeze(0))
        
        mem_input = T.cat((attn_applied, mem_input), dim=-1)
        mem_input = F.relu(self.attn_combine(mem_input))
        h_t, mem_state = self.working_memory(mem_input, mem_state)    

        state_logit = self.state_decoder(h_t)

        return h_t, mem_state, state_logit
    
    def get_init_states(self, device='cpu'):
        h0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        c0 = T.normal(0, 1, (1, 1, self.working_memory.hidden_size)).float().to(device)
        return (h0, c0) 


class DecoderRNN(nn.Module):
    def __init__(self, n_state, n_action = 4, code_size = 128, hidden_size=256):
        super(DecoderRNN, self).__init__()
        self.state_encoder = nn.Linear(n_state, code_size)
        self.action_encoder = nn.Linear(n_action, code_size)
        self.working_memory = nn.LSTM(code_size*2, hidden_size)
        self.state_decoder = nn.Linear(hidden_size, n_state)

    def forward(self, pre_state, action,  mem_state):
        state_code = self.state_encoder(pre_state.unsqueeze(0).unsqueeze(0))
        action_code = self.action_encoder(action.unsqueeze(0).unsqueeze(0))
        mem_input = T.cat((state_code, action_code), dim=-1) # (1, 1, code_size*2)  
        h_t, mem_state = self.working_memory(mem_input, mem_state)
        state_logit = self.state_decoder(h_t)

        return h_t, mem_state, state_logit

class AttnDecoderRNN(nn.Module):
    def __init__(self, length, n_state, n_action=4, code_size=128, hidden_size=256, dropout_p=.1):
        super(AttnDecoderRNN, self).__init__()
        self.state_encoder = nn.Linear(n_state, code_size)
        self.action_encoder = nn.Linear(n_action, code_size)
        self.dropout = nn.Dropout(dropout_p)
        self.working_memory = nn.LSTM(code_size*2, hidden_size)
        self.state_decoder = nn.Linear(hidden_size, n_state)
        self.attn = nn.Linear(code_size*2 + hidden_size, length)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)

    # encoder_output: (length, hidden_size)
    def forward(self, pre_state, action, mem_state, encoder_output):
        state_code = self.state_encoder(pre_state.unsqueeze(0).unsqueeze(0))
        action_code = self.action_encoder(action.unsqueeze(0).unsqueeze(0))
        mem_input = T.cat((state_code, action_code), dim=-1)
        mem_input = self.dropout(mem_input)

        attn_weight = self.attn(T.cat((mem_input[0], mem_state[0][0]), dim=1))
        attn_weight = F.softmax(attn_weight, dim=1).unsqueeze(0)
        attn_applied = T.bmm(attn_weight, encoder_output.unsqueeze(0))
        
        mem_input = T.cat((attn_applied, mem_input), dim=-1)
        mem_input = F.relu(self.attn_combine(mem_input))
        h_t, mem_state = self.working_memory(mem_input, mem_state)    

        state_logit = self.state_decoder(h_t)

        return h_t, mem_state, state_logit



