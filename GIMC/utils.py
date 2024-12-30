import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    batch_len = [len(f["sents_token"]) for f in batch]
    max_len = max([len(f["sents_token"]) for f in batch])
    input_ids = [f["sents_token"] + [0] * (max_len - len(f["sents_token"])) for f in batch]   #对齐input_ids的长度  [101, 2199, 6421, 4991, 4157, 0, 0, 0, 0]
    input_mask = [[1.0] * len(f["sents_token"]) + [0.0] * (max_len - len(f["sents_token"])) for f in batch]  #     [1, 1, 1, 1, 1, 0, 0, 0, 0]
    
    doc_event_rel_list = [f["doc_event_rel_list"] for f in batch]
    doc_phrases_new_list = [f["doc_phrases_new_list"] for f in batch]
    doc_sent_root_idx_dict = [f["doc_sent_root_idx_dict"] for f in batch]
    doc_phrases_dep_dict = [f["doc_phrases_dep_dict"] for f in batch]
    doc_phrase_same_list = [f["doc_phrase_same_list"] for f in batch]
    doc_event_dict = [f["doc_event_dict"] for f in batch]
    doc_event_phrase_dict = [f["doc_event_phrase_dict"] for f in batch]
    doc_sent_num_idx_list = [f["doc_sent_num_idx_list"] for f in batch]
    
    all_statements_token = [f["all_statements_token"] for f in batch]
    sents2statement = [f["sents2statement"] for f in batch]
    events2statement = [f["events2statement"] for f in batch]  
    state2Mstate = [f["state2Mstate"] for f in batch]
    events_sent = [f["events_sent"] for f in batch]
    events2state_pos_new_dict = [f["events2state_pos_new_dict"] for f in batch]
    casual_CL = [f["casual_CL"] for f in batch]
    file_name = [f["file_name"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    
    output = (input_ids, input_mask, doc_event_rel_list, doc_phrases_new_list, doc_sent_root_idx_dict, doc_phrases_dep_dict, doc_phrase_same_list, doc_event_dict, doc_event_phrase_dict, doc_sent_num_idx_list, all_statements_token, sents2statement, events2statement, state2Mstate, events_sent, events2state_pos_new_dict, casual_CL, batch_len, file_name)
    return output

