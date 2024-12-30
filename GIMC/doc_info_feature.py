import os
from tqdm import tqdm


def doc_info_token_process(tokenizer, doc_info_all, data_name):
    doc_info_feature_all = []
    doc_num = 0
    #doc_length_list = []
    doc_statement_num_idx_list = []   # 分词后每个陈述的长度
    for doc_info in tqdm(doc_info_all, desc='Processing to feature'):
        doc_num += 1
        sents = doc_info['sentences']
        doc_event_rel_list = doc_info['doc_event_rel_list']
        doc_event_dict = doc_info['doc_event_dict']
        doc_phrases_list = doc_info['doc_phrases_list']
        doc_phrases_dep_dict = doc_info['doc_phrases_dep_dict']
        doc_phrase_same_list = doc_info['doc_phrase_same_list']
        doc_event_phrase_dict = doc_info['doc_event_phrase_dict']
        doc_sent_root_dict = doc_info['doc_sent_root_dict']
        all_statement = doc_info['all_statement']
        sents2statement = doc_info['sents2statement']
        events2statement = doc_info['events2statement']
        events_sent = doc_info['events_sent']
        events_pos = doc_info['events_pos']
        events2state_pos_dict = doc_info['events2state_pos_dict']
        state2Mstate = doc_info['state2Mstate']
        casual_CL = doc_info['casual_CL']
        file_name = doc_info['file_name'] 

        # 对所有陈述进行分词并修改事件的位置信息
        map_statements = {}  # 所有陈述的map（范围）
        map_idx_statements = {}  # 所有陈述的map（每个下标对应哪些新下标）
        all_statements_token = []
        
        # 处理陈述的原则是，陈述包括：所有事件对对应的陈述（可能重复但一一对应）、多语扩充陈述
        # 对每一句陈述都进行分词，记录每一句分词后的map，仅对因果对比中需要标记的事件进行插入特殊符号
        for statement_id, statement in enumerate(all_statement):  
            map_statements.setdefault(statement_id, {})
            map_idx_statements.setdefault(statement_id, {})
            statement_token = []
            # 记下当前句子中事件的位置，以便插入标记
            # {0:0, 1:1, 2:7, 3:8, 4:9, 5:10, 6:11, 7:13, 8:15, 9:14, 10:15, 11:16, 12:17}
            statement_event_start = []
            statement_event_end = []
            events_id = []
            if statement_id <= (len(sents2statement)-1):  # 是原陈述 0
                for e, s in events2statement.items():
                    if statement_id == s:
                        events_id.append(e)  # 找到涉及句子的事件对

            elif statement_id > (len(sents2statement)-1):  # 是多语陈述 12
                for init_s, Mlang_s in state2Mstate.items():  # init_s = 0, Mlang_s = [12, 13, 14]
                    if statement_id in Mlang_s: 
                        for e, s in events2statement.items():
                            if s == init_s:
                                events_id.append(e)  # 找到涉及句子的事件对
            for id in events_id:
                pos0 = events_pos[id][0]
                pos1 = events_pos[id][1]
                statement_event_start.append(pos0[0])
                statement_event_start.append(pos1[0])
                statement_event_end.append(pos0[1])
                statement_event_end.append(pos1[1])

            # start = [12, 15]
            # end = [13, 16]
            for s_i, s_token in enumerate(statement):               
                s_tokens_wordpiece = tokenizer.tokenize(s_token)
                if s_i in statement_event_start:
                    s_tokens_wordpiece = ["<t>"] + s_tokens_wordpiece
                if s_i in statement_event_end:
                    s_tokens_wordpiece = ["</t>"] + s_tokens_wordpiece
                map_statements[statement_id][s_i] = len(statement_token)
                s_idx_list = [j for j in range(len(statement_token), len(statement_token)+len(s_tokens_wordpiece))]
                map_idx_statements[statement_id][s_i] = s_idx_list
                statement_token.extend(s_tokens_wordpiece)
            map_statements[statement_id][s_i+1] = len(statement_token)
            doc_statement_num_idx_list.append(len(statement_token))
            #statement_token__.append(statement_token)
            
            statement_token = tokenizer.convert_tokens_to_ids(statement_token)
            statement_token = tokenizer.build_inputs_with_special_tokens(statement_token)
            doc_statement_num_idx_list.append(len(statement_token))
            all_statements_token.append(statement_token)
        
        events2state_pos_new_dict = {}
        for event_id, state_pos_dict in events2state_pos_dict.items(): # 0: {0: ([2, 3], [13, 14]), 12: ([2, 3], [13, 14]), 13: ([2, 3], [13, 14]), 14: ([2, 3], [13, 14])}
            events2state_pos_new_dict.setdefault(event_id, {})
            for state_id, event_pos in state_pos_dict.items():  # 0: ([2, 3], [13, 14])
                map_idx = map_statements[state_id]
                init_pos1 = event_pos[0][0]  # 2
                init_pos2 = event_pos[1][0]  # 3
                init_pos3 = event_pos[0][1]  # 13
                init_pos4 = event_pos[1][1]  # 14
                pos_new = ([map_idx[init_pos1], map_idx[init_pos3]+1], [map_idx[init_pos2], map_idx[init_pos4]+1])
                events2state_pos_new_dict[event_id][state_id] = pos_new
        
        map_sents = {}  # 所有句子的map（范围）
        map_idx_sents = {}  # 所有句子的map（每个下标对应哪些新下标）
        sents_token = []
        doc_sent_num_idx_list = []
        for sent_id, sent in enumerate(sents):
            map_sents.setdefault(sent_id, {})
            map_idx_sents.setdefault(sent_id, {})
            sent_token = []
            # 记下当前句子中事件的位置，以便插入标记
            # {0:0, 1:1, 2:7, 3:8, 4:9, 5:10, 6:11, 7:13, 8:15, 9:14, 10:15, 11:16, 12:17}
            event_start = []
            event_end = []
            for event, info in doc_event_dict.items():
                if info[0] == sent_id:
                    event_start.append(info[1][0])
                    event_end.append(info[1][1]-1)
            for i, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if i in event_start:
                    tokens_wordpiece = ["<t>"] + tokens_wordpiece
                if i in event_end:
                    tokens_wordpiece = tokens_wordpiece + ["</t>"]
                map_sents[sent_id][i] = len(sent_token)
                #!!
                idx_list = [j for j in range(len(sent_token), len(sent_token)+len(tokens_wordpiece))]
                map_idx_sents[sent_id][i] = idx_list
                sent_token.extend(tokens_wordpiece)
            map_sents[sent_id][i+1] = len(sent_token)
            #map_idx_sents[sent_id][i+1] = len(sent_token)
            doc_sent_num_idx_list.append(len(sent_token))
            sents_token.extend(sent_token)

        # 修改事件信息 
        # doc_event_dict
        for event, info in doc_event_dict.items():
            map_sent = map_sents[info[0]]
            #print('map_sent:', map_sent)
            #print('=================================================')
            init_pos1 = doc_event_dict[event][1][0]
            #print('init_pos1:', init_pos1)
            #print('=================================================')
            doc_event_dict[event][1][0] = map_sent[init_pos1]
            init_pos2 = doc_event_dict[event][1][1]
            #print('init_pos2:', init_pos2)
            #print('=================================================')
            doc_event_dict[event][1][1] = map_sent[init_pos2]

        # 修改每个phrase的位置信息列表
        # doc_phrases_list
        doc_phrases_new_list = []
        for phrase in doc_phrases_list:
            map_idx_sent = map_idx_sents[phrase[0]]
            new_pos = []
            for idx in phrase[1]:
                new_pos.extend(map_idx_sent[idx])
            doc_phrases_new_list.append((phrase[0], new_pos, phrase[2]))

        doc_sent_root_idx_dict = {}
        #print('doc_sent_root_dict:', doc_sent_root_dict)
        for sent_id, root in doc_sent_root_dict.items():
            map_idx_sent = map_idx_sents[sent_id]
            doc_sent_root_idx_dict.setdefault(sent_id, [])
            new_pos_root = []
            #print('root:', root)
            for idx in root[1][0]:
                new_pos_root.extend(map_idx_sent[idx])
            doc_sent_root_idx_dict[sent_id].append(root[1][2])
            doc_sent_root_idx_dict[sent_id].append(new_pos_root)
            doc_sent_root_idx_dict[sent_id].append(root[1][1])

        sents_token = tokenizer.convert_tokens_to_ids(sents_token)
        sents_token = tokenizer.build_inputs_with_special_tokens(sents_token)

        feature = {'sents_token': sents_token,    # 文档分词后转为id的列表
                    'doc_event_rel_list': doc_event_rel_list,    # 因果对数据
                    'doc_phrases_new_list': doc_phrases_new_list,   # 文档中phrase信息列表
                    'doc_sent_root_idx_dict': doc_sent_root_idx_dict,   # 每句话根节点信息
                    'doc_phrases_dep_dict': doc_phrases_dep_dict,   # 文档中phrase之间的依赖关系
                    'doc_phrase_same_list': doc_phrase_same_list,   # 文档中相同的phrase列表
                    'doc_event_dict': doc_event_dict,   # 文档中事件信息列表
                    'doc_event_phrase_dict': doc_event_phrase_dict,   # 事件和对应的phrase
                    'doc_sent_num_idx_list': doc_sent_num_idx_list,   # 文档中每个分词后的句子长度
                    'all_statements_token': all_statements_token,
                    'sents2statement': sents2statement,   
                    'events2statement': events2statement, 
                    'state2Mstate': state2Mstate,
                    'events_sent': events_sent,  
                    'events2state_pos_new_dict': events2state_pos_new_dict,
                    'casual_CL': casual_CL,  
                    'file_name': file_name,
                    }
        doc_info_feature_all.append(feature)
    doc_statement_num_idx_list.sort()
    print('after_tokenize_max_length_{}:'.format(data_name), doc_statement_num_idx_list[-10:])  # 分词后的长度

    print("====================== of documents {}. ======================".format(doc_num))
    print("====================== Processing file... Done! ======================")
    #print("====================== File processed {}. ======================".format(len(doc_info_feature_all)))
    print('\n')
    return doc_info_feature_all
