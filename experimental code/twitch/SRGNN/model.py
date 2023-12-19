import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden




class SequenceModel(nn.Module):
    def __init__(self,opt, num_games, num_durations, num_intervals, game_embed_size, duration_embed_size, interval_embed_size, hidden_size):
        super(SequenceModel, self).__init__()
        self.num_games = num_games
        self.num_durations = num_durations
        self.num_intervals = num_intervals
        self.game_embed_size = game_embed_size
        self.duration_embed_size = duration_embed_size
        self.interval_embed_size = interval_embed_size
        self.hidden_size = hidden_size
        
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size*3, self.hidden_size*3, bias=True)
        
        self.linear_two = nn.Linear(self.hidden_size*3, self.hidden_size*3, bias=True)
     
        self.linear_three = nn.Linear(self.hidden_size*3, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 6, self.hidden_size, bias=True)

        self.linear_one1 = nn.Linear(self.hidden_size*3, self.hidden_size*3, bias=True)
        
        self.linear_two1 = nn.Linear(self.hidden_size*3, self.hidden_size*3, bias=True)
     
        self.linear_three1 = nn.Linear(self.hidden_size*3, 1, bias=False)
        self.linear_transform1 = nn.Linear(self.hidden_size * 6, self.hidden_size, bias=True)

        self.linear_one2 = nn.Linear(self.hidden_size*3, self.hidden_size*3, bias=True)
        
        self.linear_two2 = nn.Linear(self.hidden_size*3, self.hidden_size*3, bias=True)
     
        self.linear_three2 = nn.Linear(self.hidden_size*3, 1, bias=False)
        self.linear_transform2 = nn.Linear(self.hidden_size * 6, self.hidden_size, bias=True)
        

        self.dropout=0.05
 

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

        
        self.game_embedding=nn.Embedding(self.num_games, self.game_embed_size)
        self.duration_embedding=nn.Embedding(self.num_durations, self.duration_embed_size)
        self.interval_embedding=nn.Embedding(self.num_intervals, self.interval_embed_size)
       
        self.game_ggnn = GNN(self.game_embed_size,step=opt.step)
        self.duration_ggnn = GNN(self.duration_embed_size,step=opt.step)
        self.interval_ggnn = GNN( self.interval_embed_size,step=opt.step)


    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    
    def compute_scores_game(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
 
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
      
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
      
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.game_embedding.weight[1:]  # n_nodes x latent_size
   
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def compute_scores_duration(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one1(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two1(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three1(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform1(torch.cat([a, ht], 1))
        b = self.duration_embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
  
        return scores
    def compute_scores_interval(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one2(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two2(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three2(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform2(torch.cat([a, ht], 1))
        b = self.interval_embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
     
        return scores
    
    
    def forward(self, games, durations, intervals, A_game, A_duration, A_interval,alias_inputs_game,alias_inputs_duration,alias_inputs_interval):
        
        game_embed= self.game_embedding(games)
        game_embed = self.game_ggnn(A_game,game_embed)

       
        
        duration_embed= self.duration_embedding(durations)
        duration_embed = self.duration_ggnn(A_duration,duration_embed)
    
       
        interval_embed= self.interval_embedding(intervals)
        interval_embed = self.interval_ggnn(A_interval,interval_embed)

        
        get_game = lambda index1: game_embed[index1][alias_inputs_game[index1]]
        #取出每個序列包含的item對應的embedding

        
        
        get_duration = lambda index2: duration_embed[index2][alias_inputs_duration[index2]]    
       
        get_interval = lambda index3: interval_embed[index3][alias_inputs_interval[index3]]

        seq_hidden_game = torch.stack([get_game(a) for a in torch.arange(len(alias_inputs_game)).long()])
        seq_hidden_duration = torch.stack([get_duration(ai1) for ai1 in torch.arange(len(alias_inputs_duration)).long()])
        seq_hidden_interval = torch.stack([get_interval(ai2) for ai2 in torch.arange(len(alias_inputs_interval)).long()])
        

        sequence_embed = torch.cat((seq_hidden_game,seq_hidden_duration,seq_hidden_interval),dim=2) 

  
        return sequence_embed

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    
    alias_inputs_game, A_game, items_game, mask_game, targets_game= data.get_slice(i)
    alias_inputs_duration, A_duration, items_duration, mask_duration, targets_duration= data.get_slice1(i)
    alias_inputs_interval, A_interval, items_interval, mask_interval, targets_interval= data.get_slice2(i)
    alias_inputs_game = torch.Tensor(alias_inputs_game).long()
    items_game = torch.Tensor(items_game).long()
    A_game = torch.Tensor(A_game).float()
    mask_game = torch.Tensor(mask_game).long()

    alias_inputs_duration = torch.Tensor(alias_inputs_duration).long()
    items_duration = torch.Tensor(items_duration).long()
    A_duration = torch.Tensor(A_duration).float()
    mask_duration = torch.Tensor(mask_duration).long()

    alias_inputs_interval = torch.Tensor(alias_inputs_interval).long()
    items_interval = torch.Tensor(items_interval).long()
    A_interval = torch.Tensor(A_interval).float()
    mask_interval = torch.Tensor(mask_interval).long()


    hidden = model(items_game,items_duration,items_interval, A_game, A_duration,A_interval,alias_inputs_game,alias_inputs_duration,alias_inputs_interval)
  
    get_game = lambda i: hidden[i][alias_inputs_game[i]]
    
    get_duration = lambda i: hidden[i][alias_inputs_duration[i]]
   
    get_interval = lambda i: hidden[i][alias_inputs_interval[i]]


    seq_hidden_game = torch.stack([get_game(i) for i in torch.arange(len(alias_inputs_game)).long()])
    
  
  
    seq_hidden_duration = torch.stack([get_duration(i) for i in torch.arange(len(alias_inputs_duration)).long()])

   
 
    seq_hidden_interval = torch.stack([get_interval(i) for i in torch.arange(len(alias_inputs_interval)).long()])

    return targets_game,targets_duration,targets_interval ,model.compute_scores_game(seq_hidden_game, mask_game),model.compute_scores_duration(seq_hidden_duration, mask_duration),model.compute_scores_interval(seq_hidden_interval, mask_interval)
   

def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)



    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets_game,targets_duration ,targets_interval ,scores_game, scores_duration , scores_interval  = forward(model, i, train_data) 
        targets_game = torch.Tensor(targets_game).long()
        targets_duration = torch.Tensor(targets_duration).long()
        targets_interval = torch.Tensor(targets_interval).long()

        loss_game = model.loss_function(scores_game,  targets_game - 1)
        
        loss_duration = model.loss_function(scores_duration,  targets_duration - 1)

        loss_interval= model.loss_function(scores_interval,  targets_interval - 1)
      
        loss = 0.7*loss_game + 0.15*loss_duration + 0.15*loss_interval

        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit_game3,hit_game5,hit_game10, mrr_game3,mrr_game5,mrr_game10,ndcg_game3,ndcg_game5,ndcg_game10,hit_duration1, hit_duration2,hit_duration3,hit_duration5, mrr_duration2,mrr_duration3,mrr_duration5,ndcg_duration2,ndcg_duration3,ndcg_duration5,hit_interval1,hit_interval2,hit_interval3,hit_interval5, mrr_interval2, mrr_interval3, mrr_interval5,ndcg_interval2,ndcg_interval3,ndcg_interval5= [],[],[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets_game,targets_duration ,targets_interval ,scores_game, scores_duration , scores_interval  = forward(model, i, test_data)
       

        sub_scores_game3 = scores_game.topk(3)[1]
        sub_scores_game3 = trans_to_cpu(sub_scores_game3).detach().numpy()
        
        sub_scores_game5 = scores_game.topk(5)[1]
        sub_scores_game5 = trans_to_cpu(sub_scores_game5).detach().numpy()
      

        sub_scores_game10 = scores_game.topk(10)[1]
        sub_scores_game10 = trans_to_cpu(sub_scores_game10).detach().numpy()

        sub_scores_duration1 = scores_duration.topk(1)[1]
        sub_scores_duration1 = trans_to_cpu(sub_scores_duration1).detach().numpy()
        
        sub_scores_duration2 = scores_duration.topk(2)[1]
        sub_scores_duration2 = trans_to_cpu(sub_scores_duration2).detach().numpy()
        
        sub_scores_duration3 = scores_duration.topk(3)[1]
        sub_scores_duration3 = trans_to_cpu(sub_scores_duration3).detach().numpy()
        

        sub_scores_duration5 = scores_duration.topk(5)[1]
        sub_scores_duration5 = trans_to_cpu(sub_scores_duration5).detach().numpy()


        sub_scores_interval1 = scores_interval.topk(1)[1]
        sub_scores_interval1 = trans_to_cpu(sub_scores_interval1).detach().numpy()
        
        sub_scores_interval2 = scores_interval.topk(2)[1]
        sub_scores_interval2 = trans_to_cpu(sub_scores_interval2).detach().numpy()
        
        sub_scores_interval3 = scores_interval.topk(3)[1]
        sub_scores_interval3 = trans_to_cpu(sub_scores_interval3).detach().numpy()
       

        sub_scores_interval5 = scores_interval.topk(5)[1]
        sub_scores_interval5 = trans_to_cpu(sub_scores_interval5).detach().numpy()


        for score_game, target_game, mask_game in zip(sub_scores_game3, targets_game, test_data.mask_game):
            
            hit_game3.append(np.isin(target_game - 1, score_game))
            if len(np.where(score_game == target_game - 1)[0]) == 0:
                mrr_game3.append(0)
                ndcg_game3.append(0)
            else:
                mrr_game3.append(1 / (np.where(score_game == target_game - 1)[0][0] + 1))
                ndcg_game3.append(1/(np.log2((np.where(score_game == target_game - 1)[0][0] + 2))))

        for score_game, target_game, mask_game in zip(sub_scores_game5, targets_game, test_data.mask_game):
            hit_game5.append(np.isin(target_game - 1, score_game))
            if len(np.where(score_game == target_game - 1)[0]) == 0:
                mrr_game5.append(0)
                ndcg_game5.append(0)
            else:
                mrr_game5.append(1 / (np.where(score_game == target_game - 1)[0][0] + 1))
                ndcg_game5.append(1/(np.log2((np.where(score_game == target_game - 1)[0][0] + 2))))

        for score_game, target_game, mask_game in zip(sub_scores_game10, targets_game, test_data.mask_game):
            hit_game10.append(np.isin(target_game - 1, score_game))
            if len(np.where(score_game == target_game - 1)[0]) == 0:
                mrr_game10.append(0)
                ndcg_game10.append(0)
            else:
                mrr_game10.append(1 / (np.where(score_game == target_game - 1)[0][0] + 1))
                ndcg_game10.append(1/(np.log2((np.where(score_game == target_game - 1)[0][0] + 2))))

        for score_duration, target_duration, mask_duration in zip(sub_scores_duration1, targets_duration, test_data.mask_duration):
            hit_duration1.append(np.isin(target_duration - 1, score_duration))
        
        for score_duration, target_duration, mask_duration in zip(sub_scores_duration2, targets_duration, test_data.mask_duration):
            hit_duration2.append(np.isin(target_duration - 1, score_duration))
            if len(np.where(score_duration == target_duration - 1)[0]) == 0:
                mrr_duration2.append(0)
                ndcg_duration2.append(0)
            else:
                mrr_duration2.append(1 / (np.where(score_duration == target_duration - 1)[0][0] + 1))
                ndcg_duration2.append(1/(np.log2((np.where(score_duration == target_duration - 1)[0][0] + 2))))

        for score_duration, target_duration, mask_duration in zip(sub_scores_duration3, targets_duration, test_data.mask_duration):
            hit_duration3.append(np.isin(target_duration - 1, score_duration))
            if len(np.where(score_duration == target_duration - 1)[0]) == 0:
                mrr_duration3.append(0)
                ndcg_duration3.append(0)
            else:
                mrr_duration3.append(1 / (np.where(score_duration == target_duration - 1)[0][0] + 1))
                ndcg_duration3.append(1/(np.log2((np.where(score_duration == target_duration - 1)[0][0] + 2))))
        
        for score_duration, target_duration, mask_duration in zip(sub_scores_duration5, targets_duration, test_data.mask_duration):
            hit_duration5.append(np.isin(target_duration - 1, score_duration))
            if len(np.where(score_duration == target_duration - 1)[0]) == 0:
                mrr_duration5.append(0)
                ndcg_duration5.append(0)
            else:
                mrr_duration5.append(1 / (np.where(score_duration == target_duration - 1)[0][0] + 1))
                ndcg_duration5.append(1/(np.log2((np.where(score_duration == target_duration - 1)[0][0] + 2))))


        for score_interval, target_interval, mask_interval in zip(sub_scores_interval1, targets_interval, test_data.mask_interval):
            hit_interval1.append(np.isin(target_interval - 1, score_interval))
        
        for score_interval, target_interval, mask_interval in zip(sub_scores_interval2, targets_interval, test_data.mask_interval):
            hit_interval2.append(np.isin(target_interval - 1, score_interval))
            if len(np.where(score_interval == target_interval - 1)[0]) == 0:
                mrr_interval2.append(0)
                ndcg_interval2.append(0)
            else:
                mrr_interval2.append(1 / (np.where(score_interval == target_interval - 1)[0][0] + 1))
                ndcg_interval2.append(1/(np.log2((np.where(score_interval == target_interval - 1)[0][0] + 2))))
        for score_interval, target_interval, mask_interval in zip(sub_scores_interval3, targets_interval, test_data.mask_interval):
            hit_interval3.append(np.isin(target_interval - 1, score_interval))
            if len(np.where(score_interval == target_interval - 1)[0]) == 0:
                mrr_interval3.append(0)
                ndcg_interval3.append(0)
            else:
                mrr_interval3.append(1 / (np.where(score_interval == target_interval - 1)[0][0] + 1))
                ndcg_interval3.append(1/(np.log2((np.where(score_interval == target_interval - 1)[0][0] + 2))))

        for score_interval, target_interval, mask_interval in zip(sub_scores_interval5, targets_interval, test_data.mask_interval):
            hit_interval5.append(np.isin(target_interval - 1, score_interval))
            if len(np.where(score_interval == target_interval - 1)[0]) == 0:
                mrr_interval5.append(0)
                ndcg_interval5.append(0)
            else:
                mrr_interval5.append(1 / (np.where(score_interval == target_interval - 1)[0][0] + 1))
                ndcg_interval5.append(1/(np.log2((np.where(score_interval == target_interval - 1)[0][0] + 2))))

    hit_duration1 = np.mean(hit_duration1) * 100
    hit_interval1 = np.mean(hit_interval1) * 100
    hit_game3 = np.mean(hit_game3) * 100
    mrr_game3 = np.mean(mrr_game3) * 100
    ndcg_game3 = np.mean(ndcg_game3) * 100
    hit_duration2 = np.mean(hit_duration2) * 100
    mrr_duration2 = np.mean(mrr_duration2) * 100
    ndcg_duration2 = np.mean(ndcg_duration2) * 100
    hit_interval2 = np.mean(hit_interval2) * 100
    mrr_interval2 = np.mean(mrr_interval2) * 100
    ndcg_interval2 = np.mean(ndcg_interval2) * 100
    



    hit_game5 = np.mean(hit_game5) * 100
    mrr_game5 = np.mean(mrr_game5) * 100
    ndcg_game5 = np.mean(ndcg_game5) * 100
    hit_duration3 = np.mean(hit_duration3) * 100
    mrr_duration3 = np.mean(mrr_duration3) * 100
    ndcg_duration3 = np.mean(ndcg_duration3) * 100
    hit_interval3 = np.mean(hit_interval3) * 100
    mrr_interval3 = np.mean(mrr_interval3) * 100
    ndcg_interval3 = np.mean(ndcg_interval3) * 100

    hit_game10 = np.mean(hit_game10) * 100
    mrr_game10= np.mean(mrr_game10) * 100
    ndcg_game10 = np.mean(ndcg_game10) * 100
    hit_duration5 = np.mean(hit_duration5) * 100
    mrr_duration5 = np.mean(mrr_duration5) * 100
    ndcg_duration5 = np.mean(ndcg_duration5) * 100
    hit_interval5 = np.mean(hit_interval5) * 100
    mrr_interval5 = np.mean(mrr_interval5) * 100
    ndcg_interval5 = np.mean(ndcg_interval5) * 100


    return hit_duration1,hit_interval1,hit_game3, mrr_game3,ndcg_game3,hit_duration2, mrr_duration2,ndcg_duration2,hit_interval2, mrr_interval2,ndcg_interval2,hit_game5, mrr_game5,ndcg_game5,hit_duration3, mrr_duration3,ndcg_duration3,hit_interval3, mrr_interval3,ndcg_interval3,hit_game10, mrr_game10,ndcg_game10,hit_duration5, mrr_duration5,ndcg_duration5,hit_interval5, mrr_interval5,ndcg_interval5





