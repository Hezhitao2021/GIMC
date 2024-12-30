import os
import os.path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import set_seed, collate_fn
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    labels = []
    for batch in dataloader:
        model.eval()
        inputs = {'input_ids_batch': batch[0].cuda(),
                    'attention_mask_batch': batch[1].cuda(),
                    'doc_event_rel_list_batch': batch[2],
                    'doc_phrases_new_list_batch': batch[3],
                    'doc_sent_root_idx_dict_batch': batch[4],
                    'doc_phrases_dep_dict_batch': batch[5],
                    'doc_phrase_same_list_batch': batch[6],
                    'doc_event_dict_batch': batch[7],
                    'doc_event_phrase_dict_batch': batch[8],
                    'doc_sent_num_idx_list_batch': batch[9],
                    'all_statements_token_batch': batch[10],
                    'sents2statement_batch': batch[11], 
                    'events2statement_bacth': batch[12],
                    'state2Mstate_bacth': batch[13],
                    'events_sent_bacth': batch[14],
                    'events2state_pos_new_dict_bacth': batch[15],
                    'event_casual_CL_batch': batch[16],
                    'batch_len': batch[17],
                    'file_name': batch[18],
                    'train_flag': 0
                    }
        with torch.no_grad():
            pred, label, _ = model(**inputs)
        if isinstance(pred, torch.Tensor):
            top_v, pred = torch.topk(pred, 1, dim=1)
            #pred = pred.cpu().numpy()
            #pred[np.isnan(pred)] = 0
            preds.append(pred[:,0])
            labels.append(label)
        else:
            continue
    labels = torch.cat(labels,dim = 0).cpu()
    preds = torch.cat(preds,dim = 0).cpu()

    target_names  = ['label 1', 'label 2']
    labels_name = [1, 2]
    re = classification_report(labels, preds, output_dict=True, target_names=target_names, labels=labels_name)
    result = re['micro avg']
    best_f1 = result['f1-score']
    pre = result['precision']
    recall = result['recall']

    print('pre={}, recall={}'.format(pre * 100, recall * 100))
    output = {
        "事件因果实验数据" : 1,
        #tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1 * 100,
    }
 
    
    return [best_f1, pre, recall], output