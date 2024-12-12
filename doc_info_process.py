import json
import os
import random
from code_switch import get_word_dict, convert_token_in_sent 
#from code_switch_word import get_word_dict, convert_token_in_sent 
from tqdm import tqdm

# 递归搜索路径
def extend_phrase(init_phrase, dep_dic):  #[1, 4]
    phrase = recur(init_phrase, dep_dic)
    new_phrases = phrase
    new = []
    for s in new_phrases:
        if s in new:
            continue
        else:
            new.append(s)
    return new

def recur(init_phrase, dep_dic):  # [1,4]
    new_phrase = []
    for s in init_phrase:
        name = '{}'.format(s)
        if name not in dep_dic.keys():
            new_phrase.append(s)
        else:
            extend_phrase = dep_dic[name]["phrase"]
            new_phrase = new_phrase + extend_phrase # [1,3,4,8,9,10]
    t = 0
    for st in new_phrase:
        if '{}'.format(st) in dep_dic.keys():
            t += 1
    if t != 0:
        return recur(new_phrase, dep_dic)+init_phrase
    else:
        return new_phrase+init_phrase   # [1,3,4,6,7,9,10]

# 将phrase转换为文本
def phrase_idx2str(phrase_idx, sent):
    string = []
    for idx in phrase_idx:
        string.append(sent["sent_str"][idx])
    str_p = ' '.join(string)
    return str_p

def statement_extend_Multilang(sents_idx, all_init_sentences, sent_phrase_idx, Mword_dict, Mlang, extend_num):
    # 输入sents_idx，返回extend_num条多语句子
    extend_statement = []
    if sents_idx[0] == sents_idx[1]:
        sent_id = sents_idx[0]
        init_sentence = all_init_sentences[sent_id]
        #print('init_sentence :', init_sentence )
        sent_phrase = sent_phrase_idx[sent_id]
        for ii in range(extend_num):
            Mlang_statement = convert_token_in_sent(init_sentence, sent_phrase, Mword_dict, Mlang)
            #print('Mlang_statement :', Mlang_statement )
            extend_statement.append(Mlang_statement)
        #print('len=1')
    elif sents_idx[0] != sents_idx[1]: # 有两句
        sent_id1 = sents_idx[0]
        sent_id2 = sents_idx[1]
        init_sentence1 = all_init_sentences[sent_id1]
        sent_phrase1 = sent_phrase_idx[sent_id1]
        init_sentence2 = all_init_sentences[sent_id2]
        sent_phrase2 = sent_phrase_idx[sent_id2]
        #print('init_sentence1:', init_sentence1)
        #print('init_sentence2:', init_sentence2)
        for ii in range(extend_num):
            Mlang_statement1 = convert_token_in_sent(init_sentence1, sent_phrase1, Mword_dict, Mlang).copy()
            Mlang_statement2 = convert_token_in_sent(init_sentence2, sent_phrase2, Mword_dict, Mlang).copy()

            Mlang_statement_cat = Mlang_statement1 + Mlang_statement2
            #print('Mlang_statement_cat:', Mlang_statement_cat)
            extend_statement.append(Mlang_statement_cat)
    return extend_statement

# 文档信息处理
# path = '/data/MECI/MECI-v0.1/causal-en/train'
def doc_phrase_info(paths, file_name):
    #print('===Doc_info_processing_{}==='.format(file_name))
    path_type = ["root", "acl", "acl:relcl", "advcl", "appos", "conj", "csubj", "dislocated", "iobj", "list", "nsubj", "nsubj:pass", "obj", "obl", "obl:loc", "obl:tmod", "obl:npmod", "parataxis", "advmod"]  #, "nmod", "nmod:poss", "nmod:tmod", "nmod:npmod"
    remove_type = ["punct"]
    doc_info_all = []
    doc_num = 0
    dict_erro = []
    if os.path.exists(paths):
        pbar = tqdm(total=len(os.listdir(paths)), desc='Doc_info_processing_{}'.format(file_name))
        for file in os.listdir(paths):
            pbar.update(1)
            file_name_split = file.split('.')
            if len(file_name_split) == 4:
                #print('path:', path)
                #print('file:', file)
                file_path = paths + '/' + file
                if os.path.exists(file_path):
                    doc_num += 1
                    test_dp = open(file_path, 'r', encoding='utf-8')
                    test_dp = json.load(test_dp)
                    doc_id = test_dp['id']
                    events = test_dp['event']
                    relations = test_dp["relation"]  # ('T0', 'T2', 1), ('T0', 'T1', 0)
                    #text_all = test_dp['text']
                    sentences = test_dp['sentence']
                    all_sentences = test_dp['sentence']

                    # 0 事件因果关系 doc_event_rel_list
                    doc_event_rel_list = []
                    for pair, relation in relations.items():
                        event_pair = pair.split('-')
                        #print('event_pair:', event_pair)
                        if relation == "CauseEffect":#  or relation == "EffectCause":
                            doc_event_rel_list.append((event_pair[0], event_pair[1], 1))
                        elif relation == "EffectCause":
                            doc_event_rel_list.append((event_pair[0], event_pair[1], 2))
                        elif relation == "NoRel":
                            doc_event_rel_list.append((event_pair[0], event_pair[1], 0))

                    deprel = test_dp["deprel"]
                    sents = test_dp["sentence"]
                    doc_dep_info = []
                    # doc_sent_num_list
                    doc_sent_num_list = []
                    for sent in sents:
                        doc_sent_num_list.append(len(sent))

                    # 分别处理每一句的依存关系
                    sent_num = len(sents)
                    for num in range(sent_num):
                        sent_dep = {}
                        dep_path = []
                        sent_dep["sent_id"] = num
                        sent_dep["sent_str"] = sents[num]
                        for rel in deprel:
                            if rel[0][0] == num:  # 当前处理的依存关系是在当前句
                                dep_path.append((rel[1][1], rel[2], rel[0][1]))
                        # 所有的依赖路径三元组
                        sent_dep["dep_paths"] = dep_path
                        for path in dep_path:
                            name = '{}'.format(path[0])
                            if name not in sent_dep.keys():  
                                sent_dep.setdefault(name, [])
                                sent_dep[name].append((path[2], path[1]))
                            else:
                                sent_dep[name].append((path[2], path[1]))
                        doc_dep_info.append(sent_dep)


                    doc_phrase_path = []
                    for sent_info in doc_dep_info:
                        sent_phrase_path = {}
                        sent_phrase_path["sent_id"] = sent_info["sent_id"]
                        sent_phrase_path["sent_str"] = sent_info["sent_str"]
                        for i in range(-1, len(sent_info["sent_str"])):
                            name = '{}'.format(i)
                            if name not in sent_info.keys():
                                continue
                                # sent_phrase_path.setdefault(name, [[i]])  # 单个词为一个节点
                            else:
                                sent_phrase_path.setdefault(name, {}) 
                                dep_rel = sent_info[name]
                                phrase = []
                                dep_path = []
                                for rel in dep_rel:   # 除独立为边的类型和标点外，其他作为phrase的字符
                                    if rel[1] not in path_type and rel[1] not in remove_type:
                                        phrase.append(rel[0])
                                    elif rel[1] in path_type:
                                        dep_path.append(rel)
                                    else:
                                        continue
                                sent_phrase_path[name]["dep_path"] = dep_path
                                sent_phrase_path[name]["phrase"] = phrase
                        doc_phrase_path.append(sent_phrase_path)

                    # 1 得到文档的phrase列表
                    doc_phrases_list = []
                    doc_sent_root_dict = {}
                    for sent_id, sent_info in enumerate(doc_phrase_path):
                        doc_sent_root_dict.setdefault(sent_id, [])
                        for i in range(-1, len(sent_info["sent_str"])):
                            #print('i', i)
                            name = '{}'.format(i)
                            if name not in sent_info.keys():
                                continue
                            elif name == '-1':
                                phrase_path = sent_info[name]
                                #print('phrase_path:', phrase_path)
                                dep_path = phrase_path["dep_path"]
                                if len(dep_path) == 0:
                                    continue
                                else:
                                    doc_sent_root_dict[sent_id].append('{}'.format(dep_path[0][0]))
                            else:
                                phrase_path = sent_info[name]
                                dep_path = phrase_path["dep_path"]
                                phrase = phrase_path["phrase"]
                                # 判断当前节点的相连边节点是否是单个词
                                for path in dep_path:
                                    if '{}'.format(path[0]) in sent_info.keys():
                                        continue
                                    elif len(path) == None:
                                        continue
                                    elif '{}'.format(path[0]) not in sent_info.keys():  # 边节点是单个词，当作一个节点
                                        phrase_str = phrase_idx2str([path[0]], sent_info)
                                        doc_phrases_list.append((sent_id, [path[0]], phrase_str, '{}'.format(path[0])))
                                        #print('single:', [path[0]])
                                # 判断当前节点保存的phrase是否完整
                                if len(phrase) == 0:
                                    phrase_str = phrase_idx2str([i], sent_info)
                                    doc_phrases_list.append((sent_id, [i], phrase_str, name))
                                    #print('[i]', [i])
                                else:
                                    new_phrase = extend_phrase(phrase, sent_info) + [i]
                                    #print('extend:', new_phrase)
                                    phrase_str = phrase_idx2str(sorted(new_phrase), sent_info)
                                    doc_phrases_list.append((sent_id, sorted(new_phrase), phrase_str, name))
                                    #print('new_phrase', new_phrase)

                    
                    for sent_id, root in doc_sent_root_dict.items():
                        for idx, phrase_tup in enumerate(doc_phrases_list):
                            if phrase_tup[0] == sent_id and phrase_tup[-1] == root[0]:
                                doc_sent_root_dict[sent_id].append((phrase_tup[1], phrase_tup[2], idx))

                    #剔除不存在的根节点，如某句为['-']，根节点保存为：2: ['0']
                    
                    del_keys = []
                    for sent_id, root in doc_sent_root_dict.items():
                        if len(root) == 1 or len(root) == 0:
                            del_keys.append(sent_id)
                    
                    if len(del_keys) != 0:
                        dict_erro.append(file)
                        for name in del_keys:
                            del doc_sent_root_dict[name]

                    # 2 对有依赖关系的两个节点设置新的序号
                    doc_phrases_dep_dict = {}
                    for sent_id, sent_dep in enumerate(doc_phrase_path):
                        doc_phrases_dep_dict.setdefault(sent_id, [])  # {0:[]}
                        for idx1, phrase_tup in enumerate(doc_phrases_list):
                            if phrase_tup[0] == sent_id: # 当前处理的phrase是第0句的
                                if phrase_tup[-1] in sent_dep:  # phrase对应的节点是发出节点
                                    dep_rel = sent_dep[phrase_tup[-1]]["dep_path"]
                                    for rel in dep_rel:
                                        end_node = '{}'.format(rel[0])
                                        for idx2, tup in enumerate(doc_phrases_list):
                                            if tup[0] == sent_id and tup[-1] == end_node:
                                                doc_phrases_dep_dict[sent_id].append((idx1, rel[1], idx2))

                    # print('doc_phrases_dep_dict:', doc_phrases_dep_dict)
                    # print('========================================================')

                    # 3 文档中相同节点（代词剔除）
                    doc_phrase_same_list = []
                    phrase_num = len(doc_phrases_list)
                    for idx1 in range(phrase_num-1):
                        for idx2 in range(idx1+1, phrase_num):
                            #if doc_phrases_list[idx1][2] == doc_phrases_list[idx2][2] and ((doc_phrases_list[idx1][2]).lower() not in exclude_str):
                            if doc_phrases_list[idx1][2] == doc_phrases_list[idx2][2]:
                                doc_phrase_same_list.append((idx1, 'same', idx2))


                    # 4 事件信息
                    # 通过事件的位置进行定位
                    # 5 事件所在phrase信息
                    # events = test_dp['event']
                    # relations = test_dp["relation"]  # ('T0', 'T2', 1), ('T0', 'T1', 0)
                    # event_list = []
                    doc_event_dict = {}
                    doc_event_phrase_dict = {}
                    for event, event_info in events.items():
                        text = event_info["text"]
                        text_split = text.split(' ')
                        tid = event_info['tid']
                        sid = event_info['sid']
                        #text_1 = ' '.join(sents[sid][tid[0]:tid[0]+len(text_split)])
                        doc_event_dict.setdefault(event, []) 
                        doc_event_dict[event].append(sid)
                        doc_event_dict[event].append([tid[0], tid[0]+len(text_split)])
                        doc_event_phrase_dict.setdefault(event, []) 
                        for idx, phrase in enumerate(doc_phrases_list):
                            if phrase[0] == sid:
                                T = 0
                                for i in range(tid[0], tid[0]+len(text_split)):
                                    if i in phrase[1]:
                                        continue
                                    else:
                                        T += 1
                                if T == 0:
                                    doc_event_phrase_dict[event].append(idx)

                    # 6 对比学习信息处理
                    # 6.1 首先构建一个陈述列表和词典
                    # sents2statement = {(0, 1): 0, (1, 2): 1, (3, 3): 2,...}
                    # 将事件对与陈述相对应  
                    # events2statement = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5}  第0个事件对对应第1个陈述，第1个事件对对应第3个陈述
                    # 陈述中事件的位置信息
                    # events_pos = [([2, 3], [13, 14]), ([1, 2], [38, 39])]
                    # 陈述包含的句子信息、因果关系
                    # events_sent: [[(0, 1), 1], [(1, 2), 1], [(3, 3), 1]]
                    
                    sents2statement = {}  # 句子与陈述的映射
                    init_statement = []  # 陈述列表
                    events_sent = []
                    events_pos = []
                    for ind, pair in enumerate(doc_event_rel_list):
                        e1, e2, re = pair[0], pair[1], pair[2]  # ('T0', 'T2', 1)
                        e1_info = doc_event_dict[e1]  # [0,[7,8]]
                        e2_info = doc_event_dict[e2]
                        # 判断两个事件所在的句子是否相同
                        sent1 = e1_info[0]  # 2
                        sent2 = e2_info[0]  # 0
                        if sent1 == sent2:
                            e1_pos = e1_info[1]
                            e2_pos = e2_info[1]
                            #events_statement1 = sentences[sent1]
                            pair_sent = (sent1, sent2)
                            events_sent.append([pair_sent, re])  # [[(0,0), 1]]
                            if pair_sent not in sents2statement.keys():
                                init_statement.append(sentences[sent1])
                                sents2statement[pair_sent] = len(init_statement)-1      
                        elif sent1 > sent2:  
                            pair_sent = (sent2, sent1)
                            events_sent.append([pair_sent, re]) # [[(0,1), 1]]
                            events_statement1 = sentences[sent2].copy()
                            events_statement2 = sentences[sent1].copy()
                            e2_len = len(events_statement1)
                            events_statement1.extend(events_statement2)
                            e2_pos = e2_info[1]
                            e1_pos = [e1_info[1][0]+e2_len, e1_info[1][1]+e2_len]
                            if pair_sent not in sents2statement.keys():
                                init_statement.append(events_statement1)
                                sents2statement[pair_sent] = len(init_statement)-1  
                        elif sent1 < sent2:  # [sent2:sent1]
                            pair_sent = (sent1, sent2)
                            events_sent.append([pair_sent, re])
                            events_statement1 = sentences[sent1].copy()
                            events_statement2 = sentences[sent2].copy()
                            e1_len = len(events_statement1)
                            events_statement1.extend(events_statement2)
                            e1_pos = e1_info[1]
                            e2_pos = [e2_info[1][0]+e1_len, e2_info[1][1]+e1_len]
                            if pair_sent not in sents2statement.keys():
                                init_statement.append(events_statement1)
                                sents2statement[pair_sent] = len(init_statement)-1
                        events_pos.append((e1_pos, e2_pos))
                    events2statement = {}  # 事件对与陈述的映射
                    for ii, events_s in enumerate(events_sent):
                        events2statement[ii] = sents2statement[events_s[0]]

                    # 6.2 扩充多语数据
                    # 将每句话的phrase信息做成词典
                    # {0:[[1, 2, 3, 4], [6], [7], [8, 9], [10], [11, 12],...], 1:[[], [],...]}
                    sent_phrase_idx = {}
                    for phrase_info in doc_phrases_list:
                        sent_key = phrase_info[0]
                        if sent_key not in sent_phrase_idx.keys():
                            sent_phrase_idx.setdefault(sent_key, [])
                        #p_sent_id = phrase_info[0]
                        p_idx = phrase_info[1]
                        sent_phrase_idx[sent_key].append(p_idx)
                    
                    # 构建多语词典
                    word_dict_dir = "/home/Dict_for_task/en2other/"
                    langs = ['da', 'es', 'tr', 'ur']
                    #word_dict_dir = "/home//Dict_for_task/tr2other/"
                    #langs = ['en', 'da', 'es', 'ur']
                    #word_dict_dir = "/home//Dict_for_task/ur2other/"
                    #langs = ['en', 'da', 'tr', 'es']


                    Mword_dicts, Mlangs = get_word_dict(word_dict_dir, langs)
                    state2Mstate = {}  # 原陈述与扩充陈述的映射
                    for sents, statement_index in sents2statement.items():  # {(0, 1): 0, (1, 2): 1, (3, 3): 2, (0, 2): 3}
                        if statement_index not in state2Mstate.keys():
                            state2Mstate.setdefault(statement_index, [])
                            # 扩充数据
                            extend_num = 1
                            extend_statement = statement_extend_Multilang(sents, all_sentences, sent_phrase_idx, Mword_dicts, Mlangs, extend_num)
                            extend_statement_idx_list = [k for k in range(len(init_statement), len(init_statement)+extend_num)]
                            state2Mstate[statement_index].extend(extend_statement_idx_list)
                            #print('extend_statement:', extend_statement)
                            init_statement += extend_statement

                    # events_pos_dict = {0:{0:([2,3], [13,14]), 12:([2,3], [13,14]), 13:([2,3], [13,14]), 14:([2,3], [13,14])}}
                    #                   {第0个事件：{在第0个陈述中的位置：([2,3], [13,14])}}
                    events2state_pos_dict = {}
                    for events_index, e_pos in enumerate(events_pos):
                        if events_index not in events2state_pos_dict.keys():
                            events2state_pos_dict.setdefault(events_index, {})
                        statement_id = events2statement[events_index]  # 0
                        Mstatement_id = state2Mstate[statement_id]  # [12, 13, 14]
                        events2state_pos_dict[events_index][statement_id] = e_pos
                        for s_id in Mstatement_id:
                            events2state_pos_dict[events_index][s_id] = e_pos

                    # 6.3 为因果事件对分配无因果事件对
                    # event_pair_sent =  [[(0,1), 0, 1], [(2), 1, 0]]   记录每个陈述涉及的句子
                    # casual_noncausal_pair: [(0,[5,8])] （第0个事件对--因果、作为负例的第5,8个事件对--无因果）
                    casual_CL = {}
                    for ind1, pair_ce in enumerate(events_sent):  # [(0, 1), 1]
                        #print('pair_ce:', pair_ce)
                        if pair_ce[1] == 1:  # 因果事件对
                            casual_CL.setdefault(ind1, [])
                            causal_relation_sents = pair_ce[0]  # 因果事件对涉及的句子 (0, 1)
                            noncausal_list = []  # 对每一个因果事件对维护一个待选择负例列表
                            for ind2, pair_e in enumerate(events_sent):  
                                if pair_e[1] == 0:   # 非因果事件对
                                    relation_sents = pair_e[0]  # 非因果事件对涉及的句子
                                    i = 0
                                    for s in relation_sents:    # 判断非因果事件对涉及的句子是否与因果对涉及的句子重叠
                                        if s in causal_relation_sents:
                                            i += 1
                                    if i == 0:  # 若没有重叠，则把当前非因果事件对存入待选负例
                                        noncausal_list.append(ind2)
                            casual_CL[ind1].extend(noncausal_list)


                    doc_info = {'sentences': sentences,    
                                'doc_event_rel_list': doc_event_rel_list,   
                                'doc_event_dict': doc_event_dict, 
                                'doc_phrases_list': doc_phrases_list,   
                                'doc_phrases_dep_dict': doc_phrases_dep_dict,   
                                'doc_phrase_same_list': doc_phrase_same_list,   
                                'doc_event_phrase_dict': doc_event_phrase_dict,   
                                'doc_sent_root_dict': doc_sent_root_dict,
                                'all_statement': init_statement,
                                'sents2statement': sents2statement,  # 句子与陈述的映射
                                'events2statement': events2statement, # 事件对与陈述的映射
                                'events_sent': events_sent,      # 事件对的句子及因果关系
                                'events_pos': events_pos,  # 事件对位置信息
                                'events2state_pos_dict': events2state_pos_dict,  # 事件对在不同陈述中的位置
                                'state2Mstate': state2Mstate,  # 原陈述与扩充陈述的映射
                                'casual_CL': casual_CL,  # 因果事件对与非因果事件对的映射
                                'file_name': file,   
                                }
                    doc_info_all.append(doc_info)
        pbar.close()
    print("====================== of documents {}. ======================".format(doc_num))
    print("====================== Loading file... Done! ======================")
    #print("====================== info processed {} =====================".format(len(doc_info_all)))
    print('Root_dict_file_erro:', dict_erro)
    print('\n')
    return doc_info_all


