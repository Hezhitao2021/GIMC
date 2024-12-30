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
                if lang == 'es':
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

def convert_token(token, word_dicts, langs):
        this_lang = random.choice(langs)
        #raw_token = token.replace("▁", "")
        if token.lower() in word_dicts[this_lang]:
            #myrandom = random.Random(42)
            token = random.choice(word_dicts[this_lang][token.lower()])
            return token
        else:
            return token

# code switch
def convert_token_in_sent(init_sentence, sent_phrase, Mword_dict, Mlang, random_rate=0.85):
    result = []
    for token in init_sentence:
        #init_sentence = token.replace("▁", "")
        if random.random() <= random_rate:
            result.append(convert_token(token, Mword_dict, Mlang))
        else:
            result.append(token)
    return deepcopy(result)
    

# sent = ['-', '10', 'May', '1991', 'Inderjit', 'Singh', 'Reyat', 'receives', 'a', 'ten', '-', 'year', 'sentence', 'after', 'being', 'convicted', 'of', 'two', 'counts', 'of', 'manslaughter', 'and', 'four', 'explosives', 'charges', 'relating', 'to', 'the', 'Narita', 'Airport', 'bombing', '.']
# phrase_idx = [[1, 2, 3], [4, 5, 6], [7], [9, 11], [8, 9, 11, 12], [13, 14, 15], [16, 17, 18, 19, 20], [19, 20], [21, 22, 23, 24], [25], [28, 29], [26, 27, 28, 29, 30]]

# word_dict_dir = "/home//MECI/Dict_for_task/en2other/"
# langs = ['da', 'es', 'tr', 'ur']
# word_dicts1, langs1 = get_word_dict(word_dict_dir, langs)

# result = convert_token_in_sent(sent, phrase_idx, word_dicts1, langs1)

# print('sent:', sent)
# print('result:', result)


