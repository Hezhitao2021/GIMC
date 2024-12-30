import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
#from model_state_GAT2ss_base import MECIModel
from model_state_GAT2ss import MECIModel
from doc_info_process import doc_phrase_info
from doc_info_feature import doc_info_token_process
from evaluation_m import evaluate
from utils import set_seed, collate_fn

# train(args, model, train_doc_feature, dev_doc_feature)
def train(args, model, train_doc_feature, en_dev_doc_feature, en_test_doc_feature, da_dev_doc_feature, da_test_doc_feature, es_dev_doc_feature, es_test_doc_feature, tr_dev_doc_feature, tr_test_doc_feature, ur_dev_doc_feature, ur_test_doc_feature):
    def finetune(train_doc_feature, en_dev_doc_feature, en_test_doc_feature, da_dev_doc_feature, da_test_doc_feature, es_dev_doc_feature, es_test_doc_feature, tr_dev_doc_feature, tr_test_doc_feature, ur_dev_doc_feature, ur_test_doc_feature, optimizer, num_epoch, num_steps, train_flag=True):
        train_dataloader = DataLoader(train_doc_feature, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        best_f1_train = 0
        best_train = []
        train_epoch = 0

        best_f1_dev_en = 0
        best_dev_en = []
        dev_epoch_en = 0
        best_f1_test_en = 0
        best_test_en = []
        test_epoch_en = 0

        best_f1_dev_da = 0
        best_dev_da = []
        dev_epoch_da = 0
        best_f1_test_da = 0
        best_test_da = []
        test_epoch_da = 0

        best_f1_dev_es = 0
        best_dev_es = []
        dev_epoch_es = 0
        best_f1_test_es = 0
        best_test_es = []
        test_epoch_es = 0

        best_f1_dev_tr = 0
        best_dev_tr = []
        dev_epoch_tr = 0
        best_f1_test_tr = 0
        best_test_tr = []
        test_epoch_tr = 0

        best_f1_dev_ur = 0
        best_dev_ur = []
        dev_epoch_ur = 0
        best_f1_test_ur = 0
        best_test_ur = []
        test_epoch_ur = 0

        for epoch in train_iterator:
            model.zero_grad()
            pbar = tqdm(total=len(train_dataloader))
            for step, batch in enumerate(train_dataloader):
                model.train()
                pbar.update(1) 
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
                          'train_flag': 1
                          }
                outputs = model(**inputs)  #loss
                loss = outputs[0] 
                #loss.backward()
                if loss == None:
                    continue
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    if step % args.gradient_accumulation_steps == 0:
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        #scheduler.step()
                        model.zero_grad()
                        num_steps += 1
                #wandb.log({"loss": loss.item()}, step=num_steps)
                #if ((step + 1) == len(train_dataloader) and num_steps % args.evaluation_steps == 0):
            print('\n')
            print('--------------------------------------------epoch={}--------------------------------------------'.format(epoch+1))
            
            print('---------------train_data---------------')
            train_score, train_output = evaluate(args, model, train_doc_feature, tag="train")
            if train_score[0] >= best_f1_train:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_train = train_score[0]
                train_epoch = epoch+1
                if len(best_train) != 0:
                    _ = best_train.pop()
                    best_train.append((train_score[1], train_score[2]))
                else:
                    best_train.append((train_score[1], train_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            print(train_output)
            print('--------------------------------------------train_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_train*100, train_epoch,up_down))
            print('--------------------------------------------pre={}, recall={}-------------------------------------------  '.format(best_train[0][0]*100, best_train[0][1]*100))
            print('\n')
            
            # english
            print('---------------en_dev_data---------------')
            en_dev_score, en_dev_output = evaluate(args, model, en_dev_doc_feature, tag="dev")
            if en_dev_score[0]  >= best_f1_dev_en:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_dev_en = en_dev_score[0]
                dev_epoch_en = epoch+1
                if len(best_dev_en) != 0:
                    _ = best_dev_en.pop()
                    best_dev_en.append((en_dev_score[1], en_dev_score[2]))
                else:
                    best_dev_en.append((en_dev_score[1], en_dev_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(en_dev_output)
            print('--------------------------------------------en_dev_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_dev_en*100, dev_epoch_en,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_dev_en[0][0]*100, best_dev_en[0][1]*100))
            
            print('---------------en_test_data---------------')
            en_test_score, en_test_output = evaluate(args, model, en_test_doc_feature, tag="test")
            if en_test_score[0]  >= best_f1_test_en:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_test_en = en_test_score[0]
                test_epoch_en = epoch+1
                if len(best_test_en) != 0:
                    _ = best_test_en.pop()
                    best_test_en.append((en_test_score[1], en_test_score[2]))
                else:
                    best_test_en.append((en_test_score[1], en_test_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(en_test_output)
            print('--------------------------------------------en_test_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_test_en*100, test_epoch_en,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_test_en[0][0]*100, best_test_en[0][1]*100))
            print('\n')
            
            
            # danish
            print('---------------da_dev_data---------------')
            da_dev_score, da_dev_output = evaluate(args, model, da_dev_doc_feature, tag="dev")
            if da_dev_score[0]  >= best_f1_dev_da:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_dev_da = da_dev_score[0]
                dev_epoch_da = epoch+1
                if len(best_dev_da) != 0:
                    _ = best_dev_da.pop()
                    best_dev_da.append((da_dev_score[1], da_dev_score[2]))
                else:
                    best_dev_da.append((da_dev_score[1], da_dev_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(da_dev_output)
            print('--------------------------------------------da_dev_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_dev_da*100, dev_epoch_da,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_dev_da[0][0]*100, best_dev_da[0][1]*100))
            
            print('---------------da_test_data---------------')
            da_test_score, da_test_output = evaluate(args, model, da_test_doc_feature, tag="test")
            if da_test_score[0]  >= best_f1_test_da:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_test_da = da_test_score[0]
                test_epoch_da = epoch+1
                if len(best_test_da) != 0:
                    _ = best_test_da.pop()
                    best_test_da.append((da_test_score[1], da_test_score[2]))
                else:
                    best_test_da.append((da_test_score[1], da_test_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(da_test_output)
            print('--------------------------------------------da_test_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_test_da*100, test_epoch_da,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_test_da[0][0]*100, best_test_da[0][1]*100))
            print('\n')
            
            print('---------------es_dev_data---------------')
            es_dev_score, es_dev_output = evaluate(args, model, es_dev_doc_feature, tag="dev")
            if es_dev_score[0]  >= best_f1_dev_es:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_dev_es = es_dev_score[0]
                dev_epoch_es = epoch+1
                if len(best_dev_es) != 0:
                    _ = best_dev_es.pop()
                    best_dev_es.append((es_dev_score[1], es_dev_score[2]))
                else:
                    best_dev_es.append((es_dev_score[1], es_dev_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(es_dev_output)
            print('--------------------------------------------es_dev_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_dev_es*100, dev_epoch_es,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_dev_es[0][0]*100, best_dev_es[0][1]*100))
            
            print('---------------es_test_data---------------')
            es_test_score, es_test_output = evaluate(args, model, es_test_doc_feature, tag="test")
            if es_test_score[0]  >= best_f1_test_es:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_test_es = es_test_score[0]
                test_epoch_es = epoch+1
                if len(best_test_es) != 0:
                    _ = best_test_es.pop()
                    best_test_es.append((es_test_score[1], es_test_score[2]))
                else:
                    best_test_es.append((es_test_score[1], es_test_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(es_test_output)
            print('--------------------------------------------es_test_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_test_es*100, test_epoch_es,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_test_es[0][0]*100, best_test_es[0][1]*100))
            print('\n')
            
            print('---------------tr_dev_data---------------')
            tr_dev_score, tr_dev_output = evaluate(args, model, tr_dev_doc_feature, tag="dev")
            if tr_dev_score[0]  >= best_f1_dev_tr:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_dev_tr = tr_dev_score[0]
                dev_epoch_tr = epoch+1
                if len(best_dev_tr) != 0:
                    _ = best_dev_tr.pop()
                    best_dev_tr.append((tr_dev_score[1], tr_dev_score[2]))
                else:
                    best_dev_tr.append((tr_dev_score[1], tr_dev_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(tr_dev_output)
            print('--------------------------------------------tr_dev_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_dev_tr*100, dev_epoch_tr,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_dev_tr[0][0]*100, best_dev_tr[0][1]*100))
            
            print('---------------tr_test_data---------------')
            tr_test_score, tr_test_output = evaluate(args, model, tr_test_doc_feature, tag="test")
            if tr_test_score[0]  >= best_f1_test_tr:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_test_tr = tr_test_score[0]
                test_epoch_tr = epoch+1
                if len(best_test_tr) != 0:
                    _ = best_test_tr.pop()
                    best_test_tr.append((tr_test_score[1], tr_test_score[2]))
                else:
                    best_test_tr.append((tr_test_score[1], tr_test_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(tr_test_output)
            print('--------------------------------------------tr_test_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_test_tr*100, test_epoch_tr,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_test_tr[0][0]*100, best_test_tr[0][1]*100))
            print('\n')
            print('---------------ur_dev_data---------------')
            ur_dev_score, ur_dev_output = evaluate(args, model, ur_dev_doc_feature, tag="dev")
            if ur_dev_score[0]  >= best_f1_dev_ur:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_dev_ur = ur_dev_score[0]
                dev_epoch_ur = epoch+1
                if len(best_dev_ur) != 0:
                    _ = best_dev_ur.pop()
                    best_dev_ur.append((ur_dev_score[1], ur_dev_score[2]))
                else:
                    best_dev_ur.append((ur_dev_score[1], ur_dev_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(ur_dev_output)
            print('--------------------------------------------ur_dev_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_dev_ur*100, dev_epoch_ur,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_dev_ur[0][0]*100, best_dev_ur[0][1]*100))
            
            print('---------------ur_test_data---------------')
            ur_test_score, ur_test_output = evaluate(args, model, ur_test_doc_feature, tag="test")
            if ur_test_score[0]  >= best_f1_test_ur:
                #torch.save(model.state_dict(), '/data//MECI/main_code/save_weight/model_dict.pt')
                best_f1_test_ur = ur_test_score[0]
                test_epoch_ur = epoch+1
                if len(best_test_ur) != 0:
                    _ = best_test_ur.pop()
                    best_test_ur.append((ur_test_score[1], ur_test_score[2]))
                else:
                    best_test_ur.append((ur_test_score[1], ur_test_score[2]))
                up_down = '↑↑'
            else:
                up_down = '↓↓'
            #wandb.log(dev_output, step=num_steps)
            print(ur_test_output)
            print('--------------------------------------------ur_test_best_f1={}, best_epoch={}--------------------------------------------  {}'.format(best_f1_test_ur*100, test_epoch_ur,up_down))
            print('--------------------------------------------pre={}, recall={}------------------------------------------  '.format(best_test_ur[0][0]*100, best_test_ur[0][1]*100))
            
            print('==========================================================================================================================================================================================================================')
            scheduler.step()
            pbar.close()
        return num_steps

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps=args.adam_epsilon)
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    #finetune(args, train_doc_feature, da_dev_doc_feature, da_test_doc_feature, es_dev_doc_feature, es_test_doc_feature, tr_dev_doc_feature, tr_test_doc_feature, ur_dev_doc_feature, ur_test_doc_feature, optimizer, args.num_train_epochs, num_steps)
    finetune(train_doc_feature, en_dev_doc_feature, en_test_doc_feature, da_dev_doc_feature, da_test_doc_feature, es_dev_doc_feature, es_test_doc_feature, tr_dev_doc_feature, tr_test_doc_feature, ur_dev_doc_feature, ur_test_doc_feature, optimizer, args.num_train_epochs, num_steps)
    

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_dir", default="/data//MECI/MECI-v0.1/causal-en", type=str)
    parser.add_argument("--data_dir", default="/home//MECI-v0.1/causal-en", type=str)

    parser.add_argument("--test_en_dir", default="/home//MECI-v0.1/causal-en", type=str)
    parser.add_argument("--test_da_dir", default="/home//MECI-v0.1/causal-da", type=str)
    parser.add_argument("--test_es_dir", default="/home//MECI-v0.1/causal-es", type=str)
    parser.add_argument("--test_tr_dir", default="/home//MECI-v0.1/causal-tr", type=str)
    parser.add_argument("--test_ur_dir", default="/home//MECI-v0.1/causal-ur", type=str)

    #parser.add_argument("--transformer_type", default="bert", type=str)
    #parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    #parser.add_argument("--transformer_type", default="Maltehb/danish-bert-botxo", type=str)
    #parser.add_argument("--model_name_or_path", default="Maltehb/danish-bert-botxo", type=str)

    #parser.add_argument("--transformer_type", default='dccuchile/bert-base-spanish-wwm-cased', type=str)
    #parser.add_argument("--model_name_or_path", default='dccuchile/bert-base-spanish-wwm-cased', type=str)
    
    #parser.add_argument("--transformer_type", default="dbmdz/bert-base-turkish-cased", type=str)
    #parser.add_argument("--model_name_or_path", default="dbmdz/bert-base-turkish-cased", type=str)
    
    #parser.add_argument("--transformer_type", default="urduhack/roberta-urdu-small", type=str)
    #parser.add_argument("--model_name_or_path", default="urduhack/roberta-urdu-small", type=str)

    parser.add_argument("--transformer_type", default="bert-base-multilingual-cased", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str)
    
    #parser.add_argument("--transformer_type", default="xlm-roberta-base", type=str)
    #parser.add_argument("--model_name_or_path", default="xlm-roberta-base", type=str)
    
    parser.add_argument("--train_file", default="train", type=str)
    parser.add_argument("--dev_file", default="dev", type=str)
    parser.add_argument("--test_file", default="test", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--gcn_layer", default="3", type=str)
    #parser.add_argument("--local_rank", type=int)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=300.0, type=float, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=3,
                        help="Number of relation types in dataset.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    config.device = device

    #config.gcn_layer=args.gcn_layer

    set_seed(args)

    model = MECIModel(config, model, tokenizer, num_labels=args.num_class)
    #model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    #model.cuda()

    if args.load_path == "":  # Training
        # /data//MECI/MECI-v0.1/causal-en/train
        train_file = os.path.join(args.data_dir, args.train_file)
        train_doc_info = doc_phrase_info(train_file, file_name=args.train_file)
        train_doc_feature = doc_info_token_process(tokenizer, train_doc_info, data_name=args.train_file)
        
        # en
        en_dev_file = os.path.join(args.test_en_dir, args.dev_file)
        en_test_file = os.path.join(args.test_en_dir, args.test_file)
        en_dev_doc_info = doc_phrase_info(en_dev_file, file_name=args.dev_file)
        en_dev_doc_feature = doc_info_token_process(tokenizer, en_dev_doc_info, data_name=args.dev_file)
        en_test_doc_info = doc_phrase_info(en_test_file, file_name=args.test_file)
        en_test_doc_feature = doc_info_token_process(tokenizer, en_test_doc_info, data_name=args.test_file)


        # da
        # /data//MECI/MECI-v0.1/causal-da/dev
        # /data//MECI/MECI-v0.1/causal-da/test
        da_dev_file = os.path.join(args.test_da_dir, args.dev_file)
        da_test_file = os.path.join(args.test_da_dir, args.test_file)
        da_dev_doc_info = doc_phrase_info(da_dev_file, file_name=args.dev_file)
        da_dev_doc_feature = doc_info_token_process(tokenizer, da_dev_doc_info, data_name=args.dev_file)
        da_test_doc_info = doc_phrase_info(da_test_file, file_name=args.test_file)
        da_test_doc_feature = doc_info_token_process(tokenizer, da_test_doc_info, data_name=args.test_file)

        # # es
        es_dev_file = os.path.join(args.test_es_dir, args.dev_file)
        es_test_file = os.path.join(args.test_es_dir, args.test_file)
        es_dev_doc_info = doc_phrase_info(es_dev_file, file_name=args.dev_file)
        es_dev_doc_feature = doc_info_token_process(tokenizer, es_dev_doc_info, data_name=args.dev_file)
        es_test_doc_info = doc_phrase_info(es_test_file, file_name=args.test_file)
        es_test_doc_feature = doc_info_token_process(tokenizer, es_test_doc_info, data_name=args.test_file)

        # # tr
        tr_dev_file = os.path.join(args.test_tr_dir, args.dev_file)
        tr_test_file = os.path.join(args.test_tr_dir, args.test_file)
        tr_dev_doc_info = doc_phrase_info(tr_dev_file, file_name=args.dev_file)
        tr_dev_doc_feature = doc_info_token_process(tokenizer, tr_dev_doc_info, data_name=args.dev_file)
        tr_test_doc_info = doc_phrase_info(tr_test_file, file_name=args.test_file)
        tr_test_doc_feature = doc_info_token_process(tokenizer, tr_test_doc_info, data_name=args.test_file)

        # # ur
        ur_dev_file = os.path.join(args.test_ur_dir, args.dev_file)
        ur_test_file = os.path.join(args.test_ur_dir, args.test_file)
        ur_dev_doc_info = doc_phrase_info(ur_dev_file, file_name=args.dev_file)
        ur_dev_doc_feature = doc_info_token_process(tokenizer, ur_dev_doc_info , data_name=args.dev_file)
        ur_test_doc_info = doc_phrase_info(ur_test_file, file_name=args.test_file)
        ur_test_doc_feature = doc_info_token_process(tokenizer, ur_test_doc_info, data_name=args.test_file)

        #train(args, model, train_doc_feature, da_dev_doc_feature, da_test_doc_feature)
        train(args, model, train_doc_feature, en_dev_doc_feature, en_test_doc_feature, da_dev_doc_feature, da_test_doc_feature, es_dev_doc_feature, es_test_doc_feature, tr_dev_doc_feature, tr_test_doc_feature, ur_dev_doc_feature, ur_test_doc_feature)

        #train(args, model, train_doc_feature, da_dev_doc_feature, da_test_doc_feature, es_dev_doc_feature, es_test_doc_feature, tr_dev_doc_feature, tr_test_doc_feature, ur_dev_doc_feature, ur_test_doc_feature)
    else:  # Testing
        print('========Start here========')
        test_code_file = '/home//MECI/MECI-v0.1/causal-en/train_copy'
        test_code_info = doc_phrase_info(test_code_file, file_name='test_code')
        test_code_feature = doc_info_token_process(tokenizer, test_code_info , data_name='test_code')
    


if __name__ == "__main__":
    main()