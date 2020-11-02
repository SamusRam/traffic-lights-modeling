import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import sys
from map_traffic_lights_data import master_intersection_idx_2_tl_signal_indices
# early stopping source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
from pytorchtools import EarlyStopping
from datetime import timedelta
from typing import Dict, List
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.distributions import Weibull
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
from tqdm.auto import tqdm
import gc
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--trn-dataset-names', nargs='*', action='append')
parser.add_argument('--output-name', default='')
parser.add_argument('--val-file-name', default='tl_events_df_val.hdf5')
parser.add_argument('--gpu-i', default='0')
parser.add_argument('--intersection-i', default=0, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-workers', default=8, type=int)
args = parser.parse_args()

trn_dataset_names = args.trn_dataset_names[0]
gpu_i = args.gpu_i
intersection_idx = args.intersection_i
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
output_name = args.output_name
val_file_name = args.val_file_name
lr=6e-5
embedding_dim=64
hidden_dim=64
n_layers=2
bidirectional=True
dropout=0.2
epoch_max=5
lr_scheduler_patience=2
early_stopping_patience=4


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_i

TRAIN_INPUT_PATHS = [f'../input/{trn_name}' for trn_name in trn_dataset_names]
print('TRAIN_INPUT_PATHS', TRAIN_INPUT_PATHS)
VAL_INPUT_PATH = f'../input/{val_file_name}'
HIST_LEN_FRAMES = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tl_events_df_trn = pd.concat([pd.read_hdf(path, key='data') for path in TRAIN_INPUT_PATHS])
tl_events_df_val = pd.read_hdf(VAL_INPUT_PATH, key='data')

if 'continuous_time' not in tl_events_df_trn.columns:
    tl_events_df_trn['continuous_time'] = ((tl_events_df_trn['timestamp'].diff(1) < timedelta(seconds=0.31)) &  # observed 0.3 sec jumps (assuming it's between consec. scenes)
                                           (tl_events_df_trn['master_intersection_idx'].shift(1) == tl_events_df_trn['master_intersection_idx']))
if 'continuous_time' not in tl_events_df_val.columns:
    tl_events_df_val['continuous_time'] = ((tl_events_df_val['timestamp'].diff(1) < timedelta(seconds=0.31)) &
                                       (tl_events_df_val['master_intersection_idx'].shift(1) == tl_events_df_val['master_intersection_idx']))


def compute_last_valid_idx_for_seq(tl_events_df):
    tl_events_df['valid_hist_len'] = -1
    for row_i in tqdm(range(len(tl_events_df)), desc='Last valid...'):
        last_valid_idx = row_i
        while (row_i - last_valid_idx + 1 < HIST_LEN_FRAMES and 
               tl_events_df['continuous_time'].iloc[last_valid_idx]):
            last_valid_idx -= 1
        tl_events_df['valid_hist_len'].iloc[row_i] = row_i - last_valid_idx
        
if 'valid_hist_len' not in tl_events_df_trn.columns:
    compute_last_valid_idx_for_seq(tl_events_df_trn)
    if len(TRAIN_INPUT_PATHS) == 1:
        tl_events_df_trn.to_hdf(TRAIN_INPUT_PATHS[0], key='data')
    elif output_name != '':
        tl_events_df_trn.to_hdf(os.path.join('../input', f'{output_name}.hdf5'), key='data')
    else:
        print('Warn! Not storing the precomputed results!')
if 'valid_hist_len' not in tl_events_df_val.columns:
    compute_last_valid_idx_for_seq(tl_events_df_val)


train_vocab = dict()
term_freq = dict()

intersection_related_inputs = tl_events_df_trn.loc[tl_events_df_trn['master_intersection_idx'] == intersection_idx,
                                                   'rnn_inputs_raw'].values
for rnn_input_raw in intersection_related_inputs:
    for token, _ in rnn_input_raw:
        if token not in train_vocab:
            train_vocab[token] = len(train_vocab)
            term_freq[token] = 1
        else:
            term_freq[token] += 1


class IntersectionModel(nn.Module):
    
    def __init__(self, vocab_size,
                 intersection_tl_signals,
                 embedding_dim=256, 
                 hidden_dim=64, 
                 n_layers=1, 
                 bidirectional=True, 
                 dropout=0,
                 device='cuda:0'
                ):
        
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim) # including PAD and UNKNOWN tokens
        
        #lstm layer
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim + 5, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        # tl color classifier (0 -> red, 1 -> green)
        lstm_hidden_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        for tl_idx in intersection_tl_signals:
            setattr(self, f'fc_color_{tl_idx}', nn.Linear(lstm_hidden_dim, 1))
            getattr(self, f'fc_color_{tl_idx}').bias.data.fill_(0.0)
            # Weibull params
            setattr(self, f'fc_tte_k_{tl_idx}', nn.Linear(lstm_hidden_dim, 1))
            getattr(self, f'fc_tte_k_{tl_idx}').bias.data.fill_(3.0)
            setattr(self, f'fc_tte_lambda_{tl_idx}', nn.Linear(lstm_hidden_dim, 1))
            getattr(self, f'fc_tte_lambda_{tl_idx}').bias.data.fill_(2.0)
             
        # attempt to address instability, might be related to the instability reported https://github.com/ragulpr/wtte-rnn/blob/master/examples/keras/standalone_simple_example.ipynb
        def clip_and_replace_explosures(grad):
            grad[torch.logical_or(torch.isnan(grad), torch.isinf(grad))] = torch.tensor(0.0).to(device)
            grad = torch.clamp(grad, -0.25, 0.25)
            return grad
        
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(clip_and_replace_explosures)
                
        self.intersection_tl_signals = intersection_tl_signals
        
        # classifier activation function
        self.color_act = nn.Sigmoid()
        
        # Weibull params activation function
        self.param_act = nn.Softplus()
        
    def forward(self, tokens_seq, token_type_ohe, token_timesteps, seq_lengths):
        
        #tokens_seq = [batch size,sent_length]
        embedded = self.embedding(tokens_seq)
        #embedded = [batch size, sent_len, emb dim]       
        
        # adding token type ohe and timestep
        embedded = torch.cat((embedded, token_type_ohe, torch.unsqueeze(token_timesteps, 2)), dim=2)
      
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, batch_first=True, enforce_sorted = False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden shape = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward and backward hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) 
        else:
            hidden = hidden[-1,:,:]
            
        #hidden = [batch size, hid dim * num directions]

        #Final activation function
        tl_2_color_class = {tl_id: self.color_act(getattr(self, f'fc_color_{tl_id}')(hidden)) 
                            for tl_id in self.intersection_tl_signals}
        
        tl_2_tte_distr = dict()                    
        for tl_id in self.intersection_tl_signals:
            weibull_k = self.param_act(getattr(self, f'fc_tte_k_{tl_id}')(hidden))
            weibull_lambda = self.param_act(getattr(self, f'fc_tte_lambda_{tl_id}')(hidden))
            tl_2_tte_distr[tl_id] = Weibull(weibull_lambda, weibull_k)
        
        return tl_2_color_class, tl_2_tte_distr


class IntersectionDataset(Dataset):
    def __init__(self, 
                 tl_events_df: pd.DataFrame,
                 valid_indices: np.array,
                 train_vocab: Dict,
                 term_freq: Dict,
                 tl_signal_indices: List,
                 history_len_records: int = HIST_LEN_FRAMES,
                 min_freq=5
                 ):
        self.tl_events_df = tl_events_df
        max_events_per_timestamp = self.tl_events_df['rnn_inputs_raw'].map(len).max()
        self.history_events_max = history_len_records * max_events_per_timestamp
        self.history_len_records = history_len_records
        self.valid_indices = valid_indices
        self.vocab_term_2_idx = train_vocab
        self.vocab_term_2_freq = term_freq
        self.min_freq = min_freq
        self.UNKNOWN_TOKEN_IDX = len(self.vocab_term_2_idx)
        self.PAD_TOKEN_IDX = len(self.vocab_term_2_idx) + 1
        self.tl_signal_indices = tl_signal_indices
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index: int):
        row_i = self.valid_indices[index]                        
        valid_hist_len = min(self.history_len_records, self.tl_events_df['valid_hist_len'].iloc[row_i])
        raw_inputs_hist = self.tl_events_df['rnn_inputs_raw'].iloc[row_i - valid_hist_len + 1:row_i + 1]
        
        tokens_list, token_type_ohe_list, token_timesteps_list = [], [], []
        timestap = valid_hist_len
        for timestep_events in raw_inputs_hist:
            for token, token_type_ohe in timestep_events:
                # zero-max normalization
                token_timesteps_list.append(timestap/self.history_len_records)
                token_idx = self.vocab_term_2_idx[token] if token in self.vocab_term_2_idx and self.vocab_term_2_freq[token] >= self.min_freq else self.UNKNOWN_TOKEN_IDX
                tokens_list.append(token_idx)
                token_type_ohe_list.append(token_type_ohe)               
            timestap -= 1
        
        seq_len = len(tokens_list)
        tokens_np = np.array(tokens_list)
        token_type_ohe_np = np.array(token_type_ohe_list)
        token_timesteps_np = np.array(token_timesteps_list)
        
        # padding
        padding_len = self.history_events_max - seq_len
        tokens_np = np.concatenate((tokens_np, self.PAD_TOKEN_IDX*np.ones(padding_len))).astype(np.int) # shouldn't get to PAD_TOKEN_IDX, but just in case
        token_type_ohe_np = np.concatenate((token_type_ohe_np, np.zeros((padding_len, 4)))).astype(np.float32)
        token_timesteps_np = np.concatenate((token_timesteps_np, np.zeros(padding_len))).astype(np.float32)
        
        known_true_classes = self.tl_events_df['tl_signal_classes'].iloc[row_i]
        known_tte = self.tl_events_df['time_to_tl_change'].iloc[row_i]
        
        all_true_classes = {tl_id: np.float32(known_true_classes[tl_id]) if tl_id in known_true_classes else np.float32(0) for tl_id in self.tl_signal_indices}
        all_tte = {tl_id: np.float32(known_tte[tl_id]) if tl_id in known_tte else np.float32(99.0) for tl_id in self.tl_signal_indices}
        classes_availabilities = {tl_id: np.float32(tl_id in known_true_classes) for tl_id in self.tl_signal_indices}
        tte_availabilities = {tl_id: np.float32(tl_id in known_tte) for tl_id in self.tl_signal_indices}
        return tokens_np, token_type_ohe_np, token_timesteps_np, seq_len, all_true_classes, all_tte, classes_availabilities, tte_availabilities


def get_valid_indices(tl_events_df, history_len_records=HIST_LEN_FRAMES):
    # pd rolling accepts numbers only, we need to process series of size-2 tuples (is_non_empty, is_time_continuous)
    # let's encode is_non_empty, is_time_continuous as the 1st and the 2nd bit of int

    is_non_empty_bit = 0
    is_time_continuous_bit = 1

    def encode_len_continuity(records_len, is_continuous):
        res_int = 0
        if records_len >= 1:
            res_int += 1 << is_non_empty_bit
        if is_continuous:
            res_int += 1 << is_time_continuous_bit
        return res_int

    def decode_len_continuity(num):
        is_non_empty = bool(num & (1 << is_non_empty_bit))
        is_continuous = bool(num & (1 << is_time_continuous_bit))
        return is_non_empty, is_continuous

    is_nonempty___is_continuious_series = ((tl_events_df['rnn_inputs_raw'].map(lambda x: [len(x)]) + 
                                           tl_events_df['continuous_time'].map(lambda x: [x]))
                                           .map(lambda x: encode_len_continuity(*x))
                                           .astype(np.int))

    def is_nonempty_input_present(hist):
        # returns 1 when there's a non-empy input
        for i in range(len(hist) -1, -1, -1):
            is_non_empty, is_time_continuous = decode_len_continuity(int(hist.iloc[i]))
            if not is_time_continuous:
                return 0
            if is_non_empty:
                return 1
        return 0

    is_nonempty_input_present_series = is_nonempty___is_continuious_series.rolling(history_len_records - 1).agg(is_nonempty_input_present)
    
    valid_rows_bool = (tl_events_df['tl_signal_classes'].map(lambda x: len(x) > 0) &
                       (is_nonempty_input_present_series == 1))
    return np.arange(len(tl_events_df))[valid_rows_bool]


def get_dataloader(tl_events_df, intersection_idx, shuffle=True, train_vocab=train_vocab,
                   term_freq=term_freq, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    tl_events_df_intersection = tl_events_df[tl_events_df['master_intersection_idx'] == intersection_idx]
    tl_events_df_intersection.drop(['timestamp', 'master_intersection_idx'], axis=1, inplace=True)
    valid_indices_intersection = get_valid_indices(tl_events_df_intersection)
    dataset = IntersectionDataset(tl_events_df_intersection,
                                  valid_indices_intersection,
                                  train_vocab,
                                  term_freq,
                                      master_intersection_idx_2_tl_signal_indices[intersection_idx]
                                 )
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return dataloader

def train(dataloader_trn, dataloader_val, intersection_model, device, optimizer, lr_scheduler, early_stopping, 
          binary_crossentropy=nn.BCELoss(reduction="none"), epoch_max=15, clip_value=5):# ==== TRAIN LOOP
    for epoch in range(epoch_max):
        progress_bar = tqdm((dataloader_trn), desc=f'Epoch {epoch}')
        losses_train = []
        losses_lob_prob_train = []
        losses_bce_train = []

        for batch in progress_bar:
            tokens, token_type_ohe, token_timesteps, seq_len, all_true_classes, all_tte, classes_availabilities, tte_availabilities = batch
            # moving to GPU if available
            tokens, token_type_ohe, token_timesteps, seq_len = tokens.to(device), token_type_ohe.to(device), token_timesteps.to(device), seq_len.to(device)
            all_true_classes = {tl_i: vals.to(device) for tl_i, vals in all_true_classes.items()}
            all_tte = {tl_i: vals.to(device) for tl_i, vals in all_tte.items()}
            classes_availabilities = {tl_i: vals.to(device) for tl_i, vals in classes_availabilities.items()}
            tte_availabilities = {tl_i: vals.to(device) for tl_i, vals in tte_availabilities.items()}
            intersection_model.train()
            torch.set_grad_enabled(True)
            tl_2_color_class, tl_2_tte_distr = intersection_model(tokens, token_type_ohe, token_timesteps, seq_len)

            loss_bce = torch.tensor([0.0]).to(device)
            loss_bce_terms_count = torch.tensor([0.0]).to(device)
            for tl_id, pred_color_classes in tl_2_color_class.items():
                true_color_classes = all_true_classes[tl_id]
                bce_loss_tl = binary_crossentropy(torch.squeeze(pred_color_classes), true_color_classes)*classes_availabilities[tl_id]
                loss_bce += bce_loss_tl.sum()
                loss_bce_terms_count += classes_availabilities[tl_id].sum()
            if loss_bce_terms_count > 0:
                loss_bce /= loss_bce_terms_count

            loss_tte_log_prob = torch.tensor([0.0]).to(device)
            loss_tte_log_prob_terms_count = torch.tensor([0.0]).to(device)
            for tl_id, tte_distr in tl_2_tte_distr.items():
                true_ttes = torch.unsqueeze(all_tte[tl_id], -1)
                log_prob_all = torch.squeeze(tte_distr.log_prob(true_ttes))*tte_availabilities[tl_id]
                log_prob_all[torch.logical_or(torch.isnan(log_prob_all), torch.isinf(log_prob_all))] = torch.tensor(0.0).to(device)
                loss_tte_log_prob -= log_prob_all.sum()
                loss_tte_log_prob_terms_count += tte_availabilities[tl_id].sum()
            if loss_tte_log_prob_terms_count > 0:
                loss_tte_log_prob /= loss_tte_log_prob_terms_count 

            loss = loss_bce + loss_tte_log_prob

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(intersection_model.parameters(), clip_value)
            optimizer.step()

            losses_train.append(loss.item())
            losses_lob_prob_train.append(loss_tte_log_prob.item())
            losses_bce_train.append(loss_bce.item())
            progress_bar.set_description(f"Ep. {epoch}, loss: {loss.item():.2f} (bce: {loss_bce.item():.2f}, log prob: {loss_tte_log_prob.item():.3f})")
        print(f"Avg train loss: {np.mean(losses_train):.5f} (bce: {np.mean(losses_bce_train):.5f}, log prob: {np.mean(losses_lob_prob_train):.5f})")

        intersection_model.eval()
        losses_val_all = []
        losses_val_lob_prob_train = []
        losses_val_bce_train = []
        for batch in tqdm(dataloader_val, desc='Validation..'):
            tokens, token_type_ohe, token_timesteps, seq_len, all_true_classes, all_tte, classes_availabilities, tte_availabilities = batch
            tokens, token_type_ohe, token_timesteps, seq_len = tokens.to(device), token_type_ohe.to(device), token_timesteps.to(device), seq_len.to(device)
            all_true_classes = {tl_i: vals.to(device) for tl_i, vals in all_true_classes.items()}
            all_tte = {tl_i: vals.to(device) for tl_i, vals in all_tte.items()}
            classes_availabilities = {tl_i: vals.to(device) for tl_i, vals in classes_availabilities.items()}
            tte_availabilities = {tl_i: vals.to(device) for tl_i, vals in tte_availabilities.items()}
            intersection_model.train()
            torch.set_grad_enabled(True)
            tl_2_color_class, tl_2_tte_distr = intersection_model(tokens, token_type_ohe, token_timesteps, seq_len)

            loss_bce = torch.tensor([0.0]).to(device)
            loss_bce_terms_count = torch.tensor([0.0]).to(device)
            for tl_id, pred_color_classes in tl_2_color_class.items():
                true_color_classes = all_true_classes[tl_id]
                bce_loss_tl = binary_crossentropy(torch.squeeze(pred_color_classes), true_color_classes)*classes_availabilities[tl_id]
                loss_bce += bce_loss_tl.sum()
                loss_bce_terms_count += classes_availabilities[tl_id].sum()
            if loss_bce_terms_count:
                loss_bce /= loss_bce_terms_count

            loss_tte_log_prob = torch.tensor([0.0]).to(device)
            loss_tte_log_prob_terms_count = torch.tensor([0.0]).to(device)
            for tl_id, tte_distr in tl_2_tte_distr.items():
                true_ttes = torch.unsqueeze(all_tte[tl_id], -1)
                log_prob_all = torch.squeeze(tte_distr.log_prob(true_ttes)) * tte_availabilities[tl_id]
                log_prob_all[torch.logical_or(torch.isnan(log_prob_all), torch.isinf(log_prob_all))] = torch.tensor(
                    0.0).to(device)
                loss_tte_log_prob -= log_prob_all.sum()
                loss_tte_log_prob_terms_count += tte_availabilities[tl_id].sum()
            if loss_tte_log_prob_terms_count:
                loss_tte_log_prob /= loss_tte_log_prob_terms_count

            loss = loss_bce + loss_tte_log_prob/2 # to have more close scales for values of bce vs. log_prob

            losses_val_all.append(loss.item())
            losses_val_lob_prob_train.append(loss_tte_log_prob.item())
            losses_val_bce_train.append(loss_bce.item())

        loss_val_mean = np.mean(losses_val_all)
        print(f'Val loss: {loss_val_mean: .5f} (bce: {np.mean(losses_val_bce_train):.5f}, log prob: {np.mean(losses_val_lob_prob_train): .5f})')
        lr_scheduler.step(loss_val_mean)
        early_stopping(loss_val_mean, intersection_model)        
        if early_stopping.early_stop or loss_val_mean == 0:
            break


with open(f'../outputs/tl_predict_checkpoints/intersection_{intersection_idx}_fold_{gpu_i}_train_vocab.pkl', 'wb') as f:
    pickle.dump(train_vocab, f)
with open(f'../outputs/tl_predict_checkpoints/intersection_{intersection_idx}_fold_{gpu_i}_term_freq.pkl', 'wb') as f:
    pickle.dump(term_freq, f)

dataloader_trn = get_dataloader(tl_events_df_trn, intersection_idx)
dataloader_val = get_dataloader(tl_events_df_val, intersection_idx, shuffle=False)
del tl_events_df_trn, tl_events_df_val
gc.collect()
intersection_model = IntersectionModel(vocab_size=len(train_vocab),
                                       intersection_tl_signals=master_intersection_idx_2_tl_signal_indices[intersection_idx],
                                       embedding_dim=embedding_dim,
                                         hidden_dim=hidden_dim,
                                         n_layers=n_layers,
                                         bidirectional=bidirectional,
                                         dropout=dropout
                                      ).to(device)
optimizer = optim.Adam(intersection_model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
early_stopping = EarlyStopping(patience=4, verbose=True, path=f'../outputs/tl_predict_checkpoints/intersection_{intersection_idx}_fold_{gpu_i}_combined_loss_checkpoint.pt')
train(dataloader_trn, dataloader_val, intersection_model, device, optimizer, lr_scheduler, early_stopping, epoch_max=epoch_max)
