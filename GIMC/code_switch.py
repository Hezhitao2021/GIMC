import random
from copy import deepcopy

# word_dict_dir = /data/MECI/Dict_for_task/en2other/
# langs = ['da1', 'es', 'tr', 'ur']
def get_word_dict(word_dict_dir, langs: list):
    word_dicts = {lang: {} for lang in langs}
    for lang in langs:
        with open(word_dict_dir + lang + ".txt", "r",encoding='utf-8') as f:
            for line in f.readlines():
                if not line:
                    continue
                #print('line:', line)
                #print('line.split():', line.split())
                if lang == 'es' or lang == 'en':
                    line_split = line.split()
                else:
                    line_split = line.strip().split('\t')
                #print('line_split:', line_split)
                #print('lang:', lang)
                source = line_split[0] 
                target = line_split[1]
                if source.strip() == target.strip():
                    continue
                if source not in word_dicts[lang]:
                    word_dicts[lang][source] = [target]
                else:
                    word_dicts[lang][source].append(target)
    return word_dicts, langs

def convert_phrase(phrase_token, word_dicts, langs):
    this_lang = random.choice(langs) # "de"
    #print('To this_lang:', this_lang)
    #raw_token = token.replace("▁", "") # 
    raw_tokens = phrase_token
    # time = 0
    # while time < 10 and raw_token not in word_dicts[this_lang]:
    #     this_lang = random.choice(langs)
    #     time += 1
    #print('phrase_token:', phrase_token)
    if type(raw_tokens) == list:
        convert_phrase = []
        for raw_token in raw_tokens: # ['The', 'unsatisfactory', 'flight', 'plan']
            if raw_token.lower() in word_dicts[this_lang]:
                #myrandom = random.Random(42)
                token=random.choice(word_dicts[this_lang][raw_token.lower()])
            else:
                token = raw_token
            convert_phrase.append(token)
        #print('convert_phrase:', convert_phrase)
    else:
        if raw_tokens in word_dicts[this_lang]:
            #myrandom = random.Random(42)
            convert_phrase=random.choice(word_dicts[this_lang][raw_tokens])
        else:
            convert_phrase = raw_tokens
        
    return convert_phrase

def convert_token_in_sent(tokens, phrase_idx, word_dicts, langs, random_rate=0.85):
    result = []
    sent_phrase = []
    phrase_id_all = []
    phrase_idx_new = []
    for idx_list in phrase_idx:
        len_phrase = len(idx_list)
        if len_phrase != (max(idx_list)+1 - min(idx_list)):
            stand_list = [k for k in range(min(idx_list), max(idx_list)+1)]
            phrase_idx_new.append(stand_list)
        else:
            phrase_idx_new.append(idx_list)

    for i in phrase_idx_new:
        phrase_id_all.extend(i)
    #print('phrase_id_all:', phrase_id_all)
    pressed_token = []
    for s_idx, token in enumerate(tokens):
        # 当前token是否已经被处理
        if s_idx in pressed_token:
            continue
        else:
            # 当前要处理的token是否在某个phrase内
            if s_idx not in phrase_id_all:
                sent_phrase.append(token)
                pressed_token.append(s_idx)
                #print('--')
            else:
                for li in phrase_idx_new: # [8,9] [8,9,10,11,12]
                    if s_idx in li:  # s_idx=8, li=[8,9]
                        phrase_li = []
                        for j in li:
                            if j in pressed_token:
                                continue
                            else:
                                phrase_li.append(tokens[j])
                                pressed_token.append(j)
                        #print('phrase:', phrase_li)
                        sent_phrase.append(phrase_li)
    
    #print('sent_phrase:', sent_phrase)
    for phrase_token in sent_phrase:
        #print('phrase_token:', phrase_token)
        if type(phrase_token) == list:
            if random.random() <= random_rate:
                result.extend(convert_phrase(phrase_token, word_dicts, langs))
                #print('Change!')
            else:
                result.extend(phrase_token)
                #print('Not Change!')
        else:
            result.append(phrase_token)
            #print('Not Change!')
        #print('result_single:', result)
        #print('========================================================')
    if len(result) != len(tokens):
        print('=============code_switch error!============')
        print('init sent:', tokens)
        print('switch sent:', result)

    return deepcopy(result)
    

# sent = ['-', '10', 'May', '1991', 'Inderjit', 'Singh', 'Reyat', 'receives', 'a', 'ten', '-', 'year', 'sentence', 'after', 'being', 'convicted', 'of', 'two', 'counts', 'of', 'manslaughter', 'and', 'four', 'explosives', 'charges', 'relating', 'to', 'the', 'Narita', 'Airport', 'bombing', '.']
# phrase_idx = [[1, 2, 3], [4, 5, 6], [7], [9, 11], [8, 9, 11, 12], [13, 14, 15], [16, 17, 18, 19, 20], [19, 20], [21, 22, 23, 24], [25], [28, 29], [26, 27, 28, 29, 30]]

# word_dict_dir = "/data//MECI/Dict_for_task/en2other/"
# langs = ['da', 'es', 'tr', 'ur']
# word_dicts1, langs1 = get_word_dict(word_dict_dir, langs)

# result = convert_token_in_sent(sent, phrase_idx, word_dicts1, langs1)

# # print('sent:', sent)
# print('result:', result)


