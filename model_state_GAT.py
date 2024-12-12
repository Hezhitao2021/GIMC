from collections import defaultdict
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import GATv2Conv
from Focal_Loss import sigmoid_focal_loss
import transformer
import json
from seq_pro import process_long_input
import random
#from phrase_attribute import PhraseTypeEncoder

# 利用语义信息初始化，1：利用额外的PLM进行编码类型信息；2：利用同一个PLM编码类型信息
def attribute_text_process(config, phrase_attribute, type_tokenizer, type_bert):
    phrase_attribute_id_list = []
    type_list = []
    for type_, type_text in phrase_attribute.items():
        type_list.append(type_)
        type_text_token = type_tokenizer.tokenize(type_text)
        type_text_id = type_tokenizer.convert_tokens_to_ids(type_text_token)
        type_text_id = type_tokenizer.build_inputs_with_special_tokens(type_text_id)
        phrase_attribute_id_list.append(type_text_id)
    
    max_len = max([len(f) for f in phrase_attribute_id_list])
    phrase_attribute_id = [f + [0] * (max_len - len(f)) for f in phrase_attribute_id_list]   #对齐input_ids的长度  [101, 2199, 6421, 4991, 4157, 0, 0, 0, 0]
    phrase_attribute_mask = [[1.0] * len(f) + [0.0] * (max_len - len(f)) for f in phrase_attribute_id_list]  #     [1, 1, 1, 1, 1, 0, 0, 0, 0]
    #print('phrase_attribute_id_size:', len(phrase_attribute_id))
    phrase_attribute_id = torch.tensor(phrase_attribute_id, dtype=torch.long).cuda()
    phrase_attribute_mask = torch.tensor(phrase_attribute_mask, dtype=torch.float).cuda()

    phrase_attribute_sequence_output, _ = encode(type_bert, phrase_attribute_id, phrase_attribute_mask)
    
    phrase_attribute_emb = phrase_attribute_sequence_output[:,0].cuda()
    #print('phrase_attribute_emb_size:', phrase_attribute_emb.size())
    #print('type_list:', type_list)
    return phrase_attribute_emb


def encode(model, input_ids, attention_mask):

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    sequence_output = output[0]
    attention = output[-1][-1]
    
    return sequence_output, attention


# 利用语义信息初始化，1：利用额外的PLM进行编码类型信息；2：利用同一个PLM编码类型信息
class PhraseTypeEncoder(nn.Module):
    def __init__(self, config, type_bert, type_tokenizer, hidden_size):
        super(PhraseTypeEncoder, self).__init__()

        self.type_tokenizer = type_tokenizer
        self.type_bert = type_bert
        self.device = config.device
        self.phrase_att = open('/home//MECI/MPLM_main_code/phrase_attribute.json', 'r', encoding='utf-8')
        self.phrase_attribute = json.load(self.phrase_att)
        self.phrase_attribute_emb = attribute_text_process(config, self.phrase_attribute, self.type_tokenizer, self.type_bert)
        self.type_other = torch.zeros(hidden_size).unsqueeze(0).cuda()

        self.phrase_attribute_emb_all = torch.cat((self.phrase_attribute_emb, self.type_other), dim=0).cuda()
        
    def forward(self, batch_Phrase_emb, Phrase_type_ids):
        if not isinstance(Phrase_type_ids, torch.Tensor):
            Phrase_type_ids = torch.tensor(
                Phrase_type_ids, dtype=torch.long, device=batch_Phrase_emb.device, requires_grad=False
            )
        batch_Phrase_type_emb = self.phrase_attribute_emb_all[Phrase_type_ids]
        out = batch_Phrase_emb + batch_Phrase_type_emb
        out.cuda()
        return out

class MECIModel(nn.Module):
    def __init__(self, config, model, tokenizer, emb_size=768, num_labels=None):
        super(MECIModel, self).__init__()
        self.path_type = ['nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj', 'obl', 'obl:loc', 'obl:tmod', 'obl:npmod', 'dislocated', 'advcl', 'advmod', 'appos', 'acl', 'acl:relcl', 'conj', 'list', 'parataxis', 'root', 'other']  #, "nmod", "nmod:poss", "nmod:tmod", "nmod:npmod"
        self.config = config
        self.device = config.device
        self.model = model.cuda()
        self.hidden_size = config.hidden_size
        self.emb_size = emb_size
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        #self.class_weight = torch.FloatTensor([1, 5]).cuda()
        self.temperature = 1
        self.loss = nn.CrossEntropyLoss()
        self.F_Loss = sigmoid_focal_loss
        
        self.bilinear = nn.Linear(emb_size*2 , self.num_labels)
        self.num_heads=3

        self.layer_norm = transformer.LayerNorm(config.hidden_size)
        self.middle_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)    
        )
        self.middle_layer_type = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)    
        )
        self.middle_layer_gat = nn.Sequential(
            nn.Linear(config.hidden_size * (self.num_heads+1), config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)    
        )
        
        
    def contrastive_causal_loss(self, origin_embedding, positive_embedding, negative_embedding):

        positive_embedding = positive_embedding.cuda()
        negative_embedding = negative_embedding.cuda()

        positive_num = positive_embedding.shape[0]
        origin_embedding_pos = origin_embedding.repeat(positive_num, 1)
        negative_num = negative_embedding.shape[0]
        origin_embedding_neg = origin_embedding.repeat(negative_num, 1)

        l_pos = torch.bmm(origin_embedding_pos.unsqueeze(1), positive_embedding.unsqueeze(2)).reshape(positive_num, 1)
        l_neg = torch.bmm(origin_embedding_neg.unsqueeze(1), negative_embedding.unsqueeze(2)).reshape(1, negative_num)
        l_neg = l_neg.repeat(positive_num, 1)

        logits = torch.cat((l_pos, l_neg), dim=1)
        labels = torch.zeros(positive_num, dtype=torch.long).cuda()

        loss = self.loss(torch.div(logits, self.temperature), labels)

        return loss
    
    def l2_normalize(self, x, axis, epsilon: float = 1e-12):
        square_sum = torch.sum(torch.square(x), axis, True)
        x_inv_norm = torch.rsqrt(torch.max(square_sum, torch.Tensor([epsilon]).cuda()))
        # x_inv_norm = torch.rsqrt(torch.max(torch.cat((square_sum, torch.ones(square_sum.size()).cuda() * epsilon), dim=-1), dim=-1))
        return x * x_inv_norm

    def distance_cross_lang_loss(self, origin_embedding, positive_embedding):
        origin_embedding =origin_embedding
        positive_embedding = positive_embedding
        positive_num = positive_embedding.shape[0]
        #origin_emb = origin_embedding.repeat(positive_num, 1)
        dist_loss = 0
        for num in range(positive_num):
            normed_x, normed_y = self.l2_normalize(origin_embedding, axis=-1), self.l2_normalize(positive_embedding[num], axis=-1)
            dist_loss += torch.sum((normed_x - normed_y) ** 2, -1)
        return dist_loss

    def get_doc_info_emb(self, sequence_output, doc_event_rel_list, doc_sent_num_idx_list, doc_phrases_new_list, doc_event_dict, doc_sent_root_idx_dict):
        sent_num = doc_sent_num_idx_list

        doc_cls_emb = sequence_output[:,0] 

        doc_sents_token_emb = []
        doc_sents_token_emb.append(sequence_output[0, 1: sent_num[0]+1])
        if len(sent_num) >= 2:
            doc_sents_token_emb.append(sequence_output[0, sent_num[0]+1: sent_num[0]+sent_num[1]+1])
        if len(sent_num) >= 3:
            doc_sents_token_emb.append(sequence_output[0, sent_num[0]+sent_num[1]+1: sent_num[0]+sent_num[1]+sent_num[2]+1])
        if len(sent_num) >= 4:
            doc_sents_token_emb.append(sequence_output[0, sent_num[0]+sent_num[1]+sent_num[2]+1: sent_num[0]+sent_num[1]+sent_num[2]+sent_num[3]+1])
        if len(sent_num) >= 5:
            doc_sents_token_emb.append(sequence_output[0, sent_num[0]+sent_num[1]+sent_num[2]+sent_num[3]+1: sent_num[0]+sent_num[1]+sent_num[2]+sent_num[3]+sent_num[4]+1])

        # 存放每个句子的表示，此时每一句[hidden_size]（mean）
        doc_sents_emb = []
        for sent_token in doc_sents_token_emb: # sent_token: torch.Size([36, 768])
            sent_emb = sent_token.mean(dim=0)
            #print('sent_emb_size:', sent_emb.size())
            #print('===========================================================')
            doc_sents_emb.append(sent_emb)
        doc_sents_emb = torch.stack(doc_sents_emb, dim=0)
        # 融合句子位置信息
        #doc_sents_emb = self.sent_pos_encoder(doc_sents_emb, sent_pos_ids=[0,1,2,3,4])
        
        # 存放phrase的表示，此时每一个phrase [hidden_size]（mean）
        doc_phrases_emb = []
        #print('doc_phrases_new_list_len:', len(doc_phrases_new_list))
        #print('===========================================================')
        doc_sent_phrase = []  # phrase与sent_id的对应关系
        phrase_pos_id = []
        for i, phrase in enumerate(doc_phrases_new_list):
            sent_id = phrase[0]  # 0
            phrase_pos_id.append(sent_id)
            doc_sent_phrase.append((sent_id, i)) 
            phrase_emb = []
            for idx in phrase[1]:  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
                phrase_emb.append(doc_sents_token_emb[sent_id][idx])
                #print('doc_sents_token_emb[sent_id][idx]:', doc_sents_token_emb[sent_id][idx].size())
                # mean
            phrase_emb = torch.stack(phrase_emb, dim=0)  
            phrase_emb = phrase_emb.mean(dim=0)
            #print('phrase_emb:', phrase_emb.size())
            #print('===========================================================')
            doc_phrases_emb.append(phrase_emb)
        doc_phrases_emb = torch.stack(doc_phrases_emb, dim=0)
        # 融合位置信息

        doc_event_emb_dict = {}
        #print('doc_event_dict_len:', len(doc_event_dict))
        #print('===========================================================')
        for event, info in doc_event_dict.items():
            doc_event_emb_dict.setdefault(event, [])
            sent_id = info[0]
            start = info[1][0]
            end = info[1][1]
            doc_event_emb_dict[event] = doc_sents_token_emb[sent_id][start: end].mean(dim=0)

        # 存放event_pair的表示
        doc_event_node_emb = [] 
        for events in doc_event_rel_list: # ('T0', 'T2', 1)
            event1 = events[0]
            event2 = events[1]
            #event_node_emb = [doc_event_emb_dict[event1], doc_event_emb_dict[event2]]
            event_node_emb = torch.cat([doc_event_emb_dict[event1], doc_event_emb_dict[event2]])
            # 2*768 ---> 768
            event_node_emb = self.middle_layer(event_node_emb)
            doc_event_node_emb.append(event_node_emb)
        # 存在无relation的文档
        if len(doc_event_node_emb) != 0:
            doc_event_node_emb = torch.stack(doc_event_node_emb, dim=0)
        

        # 存放虚拟根节点
        doc_virtual_roots_emb = []
        for sent_id, info in doc_sent_root_idx_dict.items():
            root_phrase_idx = info[0]   # 作为根节点的phrase在list中的下标
            root_phrase = doc_phrases_emb[root_phrase_idx]
            doc_virtual_roots_emb.append(root_phrase)
        doc_virtual_roots_emb = torch.stack(doc_virtual_roots_emb, dim=0) 
        virtual_root = doc_virtual_roots_emb.mean(dim=0)  # 虚拟节点以根节点的mean为初始表示
        virtual_root = virtual_root.unsqueeze(0)

        return doc_cls_emb, doc_sents_emb, doc_phrases_emb, doc_sent_phrase, doc_event_emb_dict, doc_event_node_emb

    def forward(self,
                input_ids_batch=None,
                attention_mask_batch=None,
                doc_event_rel_list_batch=None,
                doc_phrases_new_list_batch=None,
                doc_sent_root_idx_dict_batch=None,
                doc_phrases_dep_dict_batch=None,
                doc_phrase_same_list_batch=None,
                doc_event_dict_batch=None,
                doc_event_phrase_dict_batch=None,
                doc_sent_num_idx_list_batch=None,
                all_statements_token_batch=None,
                sents2statement_batch=None,
                events2statement_bacth=None,
                state2Mstate_bacth=None,
                events_sent_bacth=None,
                events2state_pos_new_dict_bacth=None,
                event_casual_CL_batch=None,
                batch_len=None,
                file_name=None,
                train_flag=None
                ):

        #phrase_type_encode = PhraseTypeEncoder(self.config, self.model, self.tokenizer, self.hidden_size)
        # 一个batch的文档
        doc_event_pair_list = []  
        doc_cls_emb_list = []
        labels = []
        SC_loss = 0 # Sentence-level causal loss
        SiEC_loss = 0 # Signal-level event causal loss
        SiCC_loss = 0 # Signal-level context causal loss
        SnonCl_loss = 0 # Sentence-level noncausal cross-lang loss

        for input_ids, attention_mask, doc_sent_num_idx_list, doc_phrases_new_list, doc_event_dict, doc_sent_root_idx_dict, doc_event_rel_list, doc_phrases_dep_dict, doc_phrase_same_list, \
                doc_event_phrase_dict, all_statements_token, sents2statement, events2statement, state2Mstate, events_sent, events2state_pos_new_dict, event_casual_CL in zip(input_ids_batch, attention_mask_batch, doc_sent_num_idx_list_batch, doc_phrases_new_list_batch, doc_event_dict_batch, doc_sent_root_idx_dict_batch, doc_event_rel_list_batch,\
                    doc_phrases_dep_dict_batch, doc_phrase_same_list_batch, doc_event_phrase_dict_batch, all_statements_token_batch, sents2statement_batch, events2statement_bacth, state2Mstate_bacth, events_sent_bacth, events2state_pos_new_dict_bacth, event_casual_CL_batch):
                
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

            
            start_tokens = [self.config.cls_token_id]
            end_tokens = [self.config.sep_token_id]
            sequence_output, _ = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

            # 按照位置信息获取各部分表示
            doc_cls_emb, doc_sents_emb, doc_phrases_emb, doc_sent_phrase, doc_event_emb_dict, doc_event_node_emb = \
                self.get_doc_info_emb(sequence_output, doc_event_rel_list, doc_sent_num_idx_list, doc_phrases_new_list, doc_event_dict, doc_sent_root_idx_dict)

            # 若当前文档没有标注关系，则跳过
            if len(doc_event_node_emb) == 0:
                continue
            else:
                # 无mask
                start_tokens = [self.config.cls_token_id]
                end_tokens = [self.config.sep_token_id]
                max_len_statement1 = max([len(f) for f in all_statements_token])
                all_statements_id1 = [f + [0] * (max_len_statement1 - len(f)) for f in all_statements_token]   
                all_statements_mask1 = [[1.0] * len(f) + [0.0] * (max_len_statement1 - len(f)) for f in all_statements_token]  
                all_statements_id1 = torch.tensor(all_statements_id1, dtype=torch.long).cuda()
                all_statements_mask1 = torch.tensor(all_statements_mask1, dtype=torch.float).cuda()
                all_statements_seq1, _ = process_long_input(self.model, all_statements_id1, all_statements_mask1, start_tokens, end_tokens)
                all_statements_CLS = all_statements_seq1[:, 0].cuda()

                if train_flag == 1:
                    if len(event_casual_CL) != 0: # 有因果对
                        causal_statement_list = []
                        for ca_event, nonca_event_list in event_casual_CL.items():  # 0: [6, 7, 9, 15, 19, 21]
                            if len(nonca_event_list) == 0:
                                continue
                            else:
                                #print('===========================loss1============================')
                                ca_event_init_state_id = events2statement[ca_event]   # event:0, init_state: 0
                                causal_statement_list.append(ca_event_init_state_id)  # 记录因果陈述下标
                                # 初始
                                origin_embedding = all_statements_CLS[ca_event_init_state_id].cuda()

                                ### Sentence-level causal loss
                                # 负例陈述列表 
                                nonca_event_init_state_list = []
                                for nonca_id in nonca_event_list: # [6, 7, 9, 15, 19, 21]
                                    if events2statement[nonca_id] not in nonca_event_init_state_list:
                                        nonca_event_init_state_list.append(events2statement[nonca_id])  # 5,6,8,10
                                nonca_event_init_Mlang_state_list = []
                                for non_id in nonca_event_init_state_list:  
                                    nonca_event_init_Mlang_state_list.append(non_id)
                                    nonca_event_state_Mlang_list = state2Mstate[non_id]
                                    nonca_event_init_Mlang_state_list.extend(nonca_event_state_Mlang_list)
                                #print('nonca_event_init_Mlang_state_list:', nonca_event_init_Mlang_state_list)
                                # 正例陈述列表
                                ca_event_Mlang_state_list = state2Mstate[ca_event_init_state_id]    # Mlang: [12, 13, 14]
                                #print('ca_event_Mlang_state_list:', ca_event_Mlang_state_list)
                                positive_embedding = []
                                for pm_id in ca_event_Mlang_state_list:
                                    positive_embedding.append(all_statements_CLS[pm_id])
                                positive_embedding = torch.stack(positive_embedding, dim=0).cuda()

                                negative_embedding = []
                                for n_id in nonca_event_init_Mlang_state_list:
                                    negative_embedding.append(all_statements_CLS[n_id])
                                negative_embedding = torch.stack(negative_embedding, dim=0).cuda()

                                loss1 = self.contrastive_causal_loss(origin_embedding, positive_embedding, negative_embedding)  
                                SC_loss += loss1 # Sentence-level causal loss

                                event_pos_in_all_state_dict = events2state_pos_new_dict[ca_event] 
                                positive_event_embedding = []
                                for state_id, e_pos in event_pos_in_all_state_dict.items():
                                    if state_id == ca_event_init_state_id:  # 不计算origin所在陈述的事件表示
                                        continue
                                    else:
                                        e1_embeding = all_statements_seq1[state_id, e_pos[0][0]: e_pos[0][1]].mean(dim=0)
                                        e2_embeding = all_statements_seq1[state_id, e_pos[1][0]: e_pos[1][1]].mean(dim=0)
                                        events_embedding = torch.cat([e1_embeding, e2_embeding])
                                        events_embedding = self.middle_layer(events_embedding)
                                        positive_event_embedding.append(events_embedding)
                                positive_event_embedding = torch.stack(positive_event_embedding, dim=0).cuda()
                                # 负例事件编码
                                negative_event_embedding = []
                                for non_event_id in nonca_event_list:  # 0: [6, 7, 9, 15, 19, 21]
                                    event_neg_in_all_state_dict = events2state_pos_new_dict[non_event_id] # {5: ([31, 32], [7, 8]), 27: ([31, 32], [7, 8]), 28: ([31, 32], [7, 8]), 29: ([31, 32], [7, 8])}
                                    for neg_state_id, neg_e_pos in event_neg_in_all_state_dict.items():
                                        neg_e1_embeding = all_statements_seq1[neg_state_id, neg_e_pos[0][0]: neg_e_pos[0][1]].mean(dim=0)
                                        neg_e2_embeding = all_statements_seq1[neg_state_id, neg_e_pos[1][0]: neg_e_pos[1][1]].mean(dim=0)
                                        neg_events_embedding = torch.cat([neg_e1_embeding, neg_e2_embeding])
                                        neg_events_embedding = self.middle_layer(neg_events_embedding)
                                        negative_event_embedding.append(neg_events_embedding)
                                negative_event_embedding = torch.stack(negative_event_embedding, dim=0).cuda()

                                loss2 = self.contrastive_causal_loss(origin_embedding, positive_event_embedding, negative_event_embedding) 
                                SiEC_loss += loss2 # Signal-level event causal loss

                    
                    if len(event_casual_CL) == 0:
                        causal_statement_list = []
                    noncausal_state_list = []  # 无因果事件对的陈述
                    for k in range(len(state2Mstate)):
                        if k not in causal_statement_list:
                            noncausal_state_list.append(k)
                    #print('noncausal_state_list:', noncausal_state_list)
                    
                    for noncausal_state_id in noncausal_state_list:  # [3,4,5,6,7,8,9,10,11]
                        noncausal_Mlang_state_list = state2Mstate[noncausal_state_id]
                        origin_nonca_embedding = all_statements_CLS[noncausal_state_id]
                        positive_nonca_embedding  = []
                        for Mlang_id in noncausal_Mlang_state_list:
                            positive_nonca_embedding.append(all_statements_CLS[Mlang_id])
                        positive_nonca_embedding = torch.stack(positive_nonca_embedding, dim=0).cuda()
                        loss4 = self.distance_cross_lang_loss(origin_nonca_embedding, positive_nonca_embedding)
                        SnonCl_loss += loss4 # Sentence-level noncausal cross-lang loss
                    
                    if train_flag == 1:
                        if len(event_casual_CL) != 0: # 有因果对
                            # 有mask
                            # 对于因果事件对：获取origin embedding、positive context embedding、 negative context embedding
                            #   取出当前因果事件对 0 所在陈述 0，将多语陈述[12,13,14] mask event后得到正例，
                            #                                  将无句子重叠的非因果事件对陈述[5,6,8,10]及其多语陈述中随机mask 其中一个事件对作为负例
                            # 对于因果陈述，直接 mask 事件对
                            # 对于非因果陈述， 随机 mask 其中一个事件对
                            # causal_statement_list 含有因果事件对的陈述  [0,1,2] 
                            #   查看第0个陈述包含多少个因果事件对，随机选择一个
                            # noncausal_state_list  不包含因果事件对的陈述  [3,4,5,6,7,8,9,10,11] 陈述3对应的事件中随机选择一个得到事件3，得到事件3在其他陈述的位置进行mask

                            # 因果陈述处理
                            #print('===========================loss3============================')
                            state_pos_mask = [] # 为每一个非因果陈述选择一个事件进行mask
                            choiced_event_init_state_list = [] # 已经选择了因果事件的原始陈述下标
                            casual_event_list = []
                            ## 为每一个因果陈述选择一个因果对
                            for ca_event, nonca_event_list in event_casual_CL.items(): 
                                ca_event_init_state_id = events2statement[ca_event]   # event:0, init_state: 0
                                if ca_event_init_state_id not in choiced_event_init_state_list:
                                    choiced_event_init_state_list.append(ca_event_init_state_id)
                                    state_pos_mask.append((ca_event_init_state_id, ca_event))  # [0, 0] 第0个陈述 mask 第0个事件
                                    casual_event_list.append(ca_event) # 添加选中的因果事件
                            
                            ## 为每一个非因果陈述选择一个因果对
                            for nonca_sid in noncausal_state_list:
                                nonca_event_list_ = []
                                for ne, ns in events2statement.items():
                                    if ns == nonca_sid:
                                        nonca_event_list_.append(ne)
                                state_pos_mask.append((nonca_sid, random.choice(nonca_event_list_)))
                            #print('state_pos_mask:', state_pos_mask)

                            max_len_statement2 = max([len(f) for f in all_statements_token])
                            all_statements_id2 = [f + [0] * (max_len_statement2 - len(f)) for f in all_statements_token]
                            all_statements_mask2 = [[1.0] * len(f) + [0.0] * (max_len_statement2 - len(f)) for f in all_statements_token]  

                            # state_pos_mask = [(0,0), (1,1), (2,2), (3,3), (4,5), (5, 21), ...]
                            for s_p in state_pos_mask:
                                state_index = s_p[0] # 第0个陈述
                                event_index = s_p[1] # 第0个事件
                                event2state_mask_pos_dict = events2state_pos_new_dict[event_index]
                                for mask_state_id, mask_pos in event2state_mask_pos_dict.items():
                                    if mask_state_id in choiced_event_init_state_list:
                                        continue
                                    else:
                                        mask_state = all_statements_mask2[mask_state_id]
                                        event_pos1 = mask_pos[0]  # [2, 3]
                                        event_pos2 = mask_pos[1]  # [13, 14]
                                        for pos1 in range(event_pos1[0], event_pos1[1]):
                                            mask_state[pos1] = 0
                                        for pos2 in range(event_pos2[0], event_pos2[1]):
                                            mask_state[pos2] = 0
                            all_statements_id2 = torch.tensor(all_statements_id2, dtype=torch.long).cuda()
                            all_statements_mask2 = torch.tensor(all_statements_mask2, dtype=torch.float).cuda()
                            #print('all_statements_mask2[-1]:', all_statements_mask2[-1])
                            all_statements_mask_seq, _ = process_long_input(self.model, all_statements_id2, all_statements_mask2, start_tokens, end_tokens)

                            all_statements_mask_CLS = all_statements_mask_seq[:, 0].cuda()

                            for ca_event, nonca_event_list in event_casual_CL.items():  # 0: [6, 7, 9, 15, 19, 21]
                                if len(nonca_event_list) == 0:
                                    continue
                                else:
                                    if ca_event in casual_event_list:
                                        #print('ca_event:', ca_event)
                                        ca_event_init_state_id = events2statement[ca_event]   # event:0, init_state: 0
                                        #causal_statement_list.append(ca_event_init_state_id)
                                        # 初始
                                        origin_embedding = all_statements_mask_CLS[ca_event_init_state_id].cuda()
                                        # 负例陈述列表 
                                        nonca_event_init_state_list = []
                                        for nonca_id in nonca_event_list: # [6, 7, 9, 15, 19, 21]
                                            if events2statement[nonca_id] not in nonca_event_init_state_list:
                                                nonca_event_init_state_list.append(events2statement[nonca_id])  # 5,6,8,10
                                        nonca_event_init_Mlang_state_list = []
                                        for non_id in nonca_event_init_state_list:  
                                            nonca_event_init_Mlang_state_list.append(non_id)
                                            nonca_event_state_Mlang_list = state2Mstate[non_id]
                                            nonca_event_init_Mlang_state_list.extend(nonca_event_state_Mlang_list)
                                        #print('nonca_event_init_Mlang_state_list:', nonca_event_init_Mlang_state_list)
                                        # 正例陈述列表
                                        ca_event_Mlang_state_list = state2Mstate[ca_event_init_state_id]    # Mlang: [12, 13, 14]
                                        #print('ca_event_Mlang_state_list:', ca_event_Mlang_state_list)

                                        positive_context_embedding = []
                                        for pm_id in ca_event_Mlang_state_list:
                                            positive_context_embedding.append(all_statements_mask_CLS[pm_id])
                                        positive_context_embedding = torch.stack(positive_context_embedding, dim=0).cuda()

                                        negative_context_embedding = []
                                        for n_id in nonca_event_init_Mlang_state_list:
                                            negative_context_embedding.append(all_statements_mask_CLS[n_id])
                                        negative_context_embedding = torch.stack(negative_context_embedding, dim=0).cuda()

                                        loss3 = self.contrastive_causal_loss(origin_embedding, positive_context_embedding, negative_context_embedding)
                                        SiCC_loss += loss3 # Signal-level context causal loss
                
                init_statement_emb = all_statements_CLS[:len(state2Mstate)].cuda()
                #print('len_init_state:', len(init_statement_emb))
                doc_cls_emb = doc_cls_emb.squeeze(0)
                    
                # 添加类型信息
                # self.path_type
                phrase_type_dict = {}
                for sent, rel in doc_phrases_dep_dict.items():
                    for r in rel:
                        phrase_i = r[2]
                        relation_i = r[1]
                        phrase_type_dict[phrase_i] = self.path_type.index(relation_i)

                for sent, root in doc_sent_root_idx_dict.items():
                    phrase_i = root[0]
                    phrase_type_dict[phrase_i] = self.path_type.index('root')
                
                sorted_type_list = sorted(phrase_type_dict.items(),key=lambda s:s[0])
                
                #print('sort_dep_dict:', sorted_type_list)

                phrase_type_list = []
                for k in range(doc_phrases_emb.size(0)):
                    n = 0
                    for p in sorted_type_list:
                        if p[0] == k:
                            n += 1
                            phrase_type_list.append(p[1])
                    if n == 0:
                        phrase_type_list.append(19)

                # 添加类型信息
                #doc_phrases_emb = phrase_type_encode(doc_phrases_emb, Phrase_type_ids=phrase_type_list).cuda()

                # 将文档内所有因果标签存入
                # 将文档表示存入
                for event_pair in doc_event_rel_list:
                    labels.append(torch.tensor(event_pair[2]))
                    doc_cls_emb_list.append(doc_cls_emb)
                
                # 用事件陈述代替句子
                node_features = [init_statement_emb, doc_phrases_emb, doc_event_node_emb]
                # node_features = [doc_sents_emb, doc_phrases_emb, doc_event_node_emb]
                
                node_features = torch.cat(node_features, dim=0)
                node_feature = init_statement_emb

                sent_num = node_feature.size(0)

                phrase_num = doc_phrases_emb.size(0)

                link_list = []
                
                # sents2statement 陈述与句子的映射

                # 1 陈述与短语
                # for sents, state in sents2statement.items():
                #     # sents = (0, 1), state = 0
                #     for tup in doc_sent_phrase: # (sent_id, i)
                #         if tup[0] in sents:
                #             if (state, tup[1]+sent_num) not in link_list:
                #                 link_list.append((state, tup[1]+sent_num))

                # 1.1 句子与短语
                # for tup in doc_sent_phrase: # (sent_id, i)
                #     link_list.append((tup[0], tup[1]+sent_num))

                # 2 短语之间依赖关系
                for sent, rel in doc_phrases_dep_dict.items():
                    for dp in rel:
                        link_list.append((dp[0]+sent_num, dp[2]+sent_num))

                # 3 相同短语
                for same in doc_phrase_same_list:
                    if (same[0]+sent_num, same[2]+sent_num) not in link_list:
                        link_list.append((same[0]+sent_num, same[2]+sent_num))

                # 4 陈述与在其中的events_node
                # events2statement 事件与陈述的映射
                # sdoc_event_dict 事件在句子中的位置
                for idx, event_node in enumerate(doc_event_rel_list):
                    # idx = 0, event_node = ('T0', 'T2', 1)
                    node_e1 = doc_event_dict[event_node[0]][0] # 第一个事件所在句子
                    node_e2 = doc_event_dict[event_node[1]][0]
                    e2 = max([node_e1, node_e2])
                    e1 = min([node_e1, node_e2])
                    for sents_, state_ in sents2statement.items():
                        if (e1, e2) == sents_:
                            if (state_, sent_num+phrase_num+idx) not in link_list:
                                link_list.append((state_, sent_num+phrase_num+idx))
                
                # 4.1 句子与在其中的events_node
                # doc_event_dict：{'T0': [0, [12, 15]], 'T1': [0, [17, 20]],
                # doc_event_rel_list：[('T0', 'T2', 1), ('T2', 'T0', 1)
                # for idx, event_node in enumerate(doc_event_rel_list):
                #     event_sent = []
                #     event1 =  event_node[0]
                #     event2 =  event_node[1]
                #     sent_id1 = doc_event_dict[event1][0]
                #     sent_id2 = doc_event_dict[event2][0]
                #     if sent_id1 not in event_sent:
                #         event_sent.append(sent_id1)
                #     if sent_id2 not in event_sent:
                #         event_sent.append(sent_id2)
                #     for sent in event_sent:
                #         # 防止重复边
                #         if (sent, sent_num+phrase_num+idx) not in link_list:
                #             link_list.append((sent, sent_num+phrase_num+idx))


                # 5 短语与其中的events_node
                # doc_event_phrase_dict：{'T0':[1,3,6]}（第0个事件在1、3、6这三个phrase中）
                for idx, event_node in enumerate(doc_event_rel_list):
                    event1 =  event_node[0]
                    event2 =  event_node[1]
                    phrase_list1 =  doc_event_phrase_dict[event1]
                    phrase_list2 =  doc_event_phrase_dict[event2]
                    for p in phrase_list1:
                        if (p+sent_num, sent_num+phrase_num+idx) not in link_list:
                            link_list.append((p+sent_num, sent_num+phrase_num+idx))
                    for h in phrase_list2:
                        if (h+sent_num, sent_num+phrase_num+idx) not in link_list:
                            link_list.append((h+sent_num, sent_num+phrase_num+idx))

                # 6 events_node 之间
                for j in range(len(doc_event_rel_list)):
                    event1 =  doc_event_rel_list[j][0]
                    event2 =  doc_event_rel_list[j][1]
                    for k in range(j+1, len(doc_event_rel_list)):
                        event_list = []
                        event3 =  doc_event_rel_list[k][0]
                        event4 =  doc_event_rel_list[k][1]
                        if event1 not in event_list:
                            event_list.append(event1)
                        if event2 not in event_list:
                            event_list.append(event2)
                        if event3 not in event_list:
                            event_list.append(event3)
                        if event4 not in event_list:
                            event_list.append(event4)
                        if len(event_list) != 4:
                            if (sent_num+phrase_num+j, sent_num+phrase_num+k) not in link_list:
                                link_list.append((sent_num+phrase_num+j, sent_num+phrase_num+k))
                
                # # 6 events_node 之间均相连
                # for j in range(len(doc_event_rel_list)):
                #     for k in range(j+1, len(doc_event_rel_list)):
                #         link_list.append((sent_num+phrase_num+j, sent_num+phrase_num+k))
                
                init_node = []
                end_node = []
                for link in link_list:
                    init_node.append(link[0])
                    end_node.append(link[1])
                
                g = dgl.graph((init_node, end_node)).to(device='cuda')
                g = dgl.add_self_loop(g)
                
                gatv2conv = GATv2Conv(self.emb_size, self.emb_size, self.num_heads).to(device='cuda')
                node_gat = gatv2conv(g, node_features)

                event_pair_gat = node_gat[sent_num+phrase_num:,:,:]

                event_pair = event_pair_gat[:,0,:]

                for i in range(1, self.num_heads):
                    event_pair = torch.cat([event_pair, event_pair_gat[:,i,:]], dim=1)

                #print('event_pair_size:', event_pair.size())
                #print('doc_event_node_emb:', doc_event_node_emb.size())

                event_pair = torch.cat([doc_event_node_emb, event_pair], dim=1)
                
                event_pair = self.middle_layer_gat(event_pair)
                #print('event_pair_size:', event_pair.size())
                
                doc_event_pair_list.append(event_pair)
        if len(doc_cls_emb_list) == 0:  # 未标注事件关系
            output = [None, labels, 'logits']
        else:
            doc_cls_emb_list = torch.stack(doc_cls_emb_list, dim=0)
            doc_cls_emb_list = self.layer_norm(doc_cls_emb_list)
            doc_event_pair_list = torch.cat(doc_event_pair_list, dim=0)
            doc_event_pair_list = self.layer_norm(doc_event_pair_list)

            labels = torch.stack(labels,dim=0).cuda()
            #print('doc_event_pair_list_size:', doc_event_pair_list.size())
            #print('doc_cls_emb_list_size:', doc_cls_emb_list.size())

            logits = self.bilinear(torch.cat([doc_event_pair_list, doc_cls_emb_list], dim=1)).cuda()
            if train_flag == 1:
                label_y = torch.zeros(len(labels), self.num_labels).cuda()
                label_y[range(label_y.shape[0]), labels] = 1
                label_y = label_y.cuda()
                F_loss = sigmoid_focal_loss(logits, label_y)
                Total_loss = SC_loss + SiEC_loss + SiCC_loss + SnonCl_loss + F_loss
                output = [Total_loss, logits, 'loss']
                #output = [F_loss, logits, 'loss']
            else:
                output = [logits, labels, 'logits']
                #print('logit:', logits)
        return output
