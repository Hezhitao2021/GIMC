from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
import torch
import torch.nn as nn

phrase_attribute = {}
phrase_attribute['nsubj'] = 'A nominal subject (nsubj) is a nominal which is the syntactic subject and the proto-agent of a clause. That is, it is in the position that passes typical grammatical test for subjecthood, and this argument is the more agentive, the do-er, or the proto-agent of the clause.'
phrase_attribute['nsubj:pass'] = 'A passive nominal subject is a noun phrase which is the syntactic subject of a passive clause.'
phrase_attribute['obj'] = 'The object of a verb is the second most core argument of a verb after the subject. Typically, it is the noun phrase that denotes the entity acted upon or which undergoes a change of state or motion (the proto-patient).'
phrase_attribute['iobj'] = 'The indirect object of a verb is any nominal phrase that is a core argument of the verb but is not its subject or (direct) object.'
phrase_attribute['csubj'] = 'A clausal subject is a clausal syntactic subject of a clause, i.e., the subject is itself a clause. The dependent is the main lexical verb or other predicate of the subject clause.'
phrase_attribute['obl'] = 'The obl relation is used for a nominal (noun, pronoun, noun phrase) functioning as a non-core (oblique) argument or adjunct. This means that it functionally corresponds to an adverbial attaching to a verb, adjective or other adverb.s'
phrase_attribute['obl:loc'] = 'The place modifier is a subtype of the obl relationship: if the modifier specifies a place, it is marked loc.'
phrase_attribute['obl:tmod'] = 'A temporal modifier is a subtype of the obl relation: if the modifier is specifying a time, it is labeled as tmod.'
phrase_attribute['obl:npmod'] = 'This relation is a subtype of the obl relation, which captures cases where a noun phrase is used as an adverbial modifier in a sentence'
phrase_attribute['dislocated'] = 'The dislocated relation is used for fronted or postposed elements that do not fulfill the usual core grammatical relations of a sentence. These elements often appear to be in the periphery of the sentence, and may be separated off with a comma intonation.'
phrase_attribute['advcl'] = 'An adverbial clause modifier is a clause which modifies a verb or other predicate (adjective, etc.), as a modifier not as a core complement. This includes things such as a temporal clause, consequence, conditional clause, purpose clause, etc.'
phrase_attribute['advmod'] = 'An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase that serves to modify a predicate or a modifier word.'
phrase_attribute['appos'] = 'An appositional modifier of a noun is a nominal immediately following the first noun that serves to define, modify, name, or describe that noun. It includes parenthesized examples, as well as defining abbreviations in one of these structures.appos is intended to be used between two nominals. In general, modulo punctuation, the two halves of an apposition can be switched. '
phrase_attribute['acl'] = 'acl stands for finite and non-finite clauses that modify a nominal. The head of the acl relation is the noun that is modified, and the dependent is the head of the clause that modifies the noun.'
phrase_attribute['acl:relcl'] = 'A relative clause modifier of a nominal is a clause that modifies the nominal, whereas the nominal is coreferential with a constituent inside the relative clause.'
phrase_attribute['conj'] = 'A conjunct is the relation between two elements connected by a coordinating conjunction, such as and, or, etc.'
phrase_attribute['list'] = 'The list relation is used for chains of comparable items. '
phrase_attribute['parataxis'] = 'The parataxis relation is a relation between a word (often the main predicate of a sentence) and other elements, such as a sentential parenthetical or a clause after a “:” or a “;”, placed side by side without any explicit coordination, subordination, or argument relation with the head word. Parataxis is a discourse-like equivalent of coordination, and so usually obeys an iconic ordering. '
phrase_attribute['root'] = 'The root grammatical relation points to the root of the sentence. '

with open('/data//MECI/MPLM_main_code/phrase_attribute.json', 'w') as f0:
    line = str(phrase_attribute)
    line = line.replace("'", '"')
    f0.writelines(line)
    f0.close()















def attribute_text_process(phrase_attribute, type_tokenizer, type_bert):
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
    phrase_attribute_id = torch.tensor(phrase_attribute_id, dtype=torch.long).to(device='cuda')
    phrase_attribute_mask = torch.tensor(phrase_attribute_mask, dtype=torch.float).to(device='cuda')

    phrase_attribute_sequence_output, _ = encode(type_bert, phrase_attribute_id, phrase_attribute_mask)
    
    phrase_attribute_emb = phrase_attribute_sequence_output[:,0].to(device='cuda')
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
    
        self.phrase_att = open('/data//MECI/da_main_code/phrase_attribute.json', 'r', encoding='utf-8')
        self.phrase_attribute = json.load(self.phrase_att)
        self.phrase_attribute_emb = attribute_text_process(self.phrase_attribute, self.type_tokenizer, self.type_bert).to(device='cuda')
        self.type_other = torch.zeros(hidden_size).unsqueeze(0).to(device='cuda')
        #print('self.phrase_attribute_emb_size:', self.phrase_attribute_emb.size())
        #print('self.type_other_size:', self.type_other.size())

        self.phrase_attribute_emb_all = torch.cat((self.phrase_attribute_emb, self.type_other), dim=0).to(device='cuda')
        #print('emb_all_size:', self.phrase_attribute_emb_all.size())
        #print('emb_all_zero:', self.phrase_attribute_emb_all[19])

    def forward(self, batch_Phrase_emb, Phrase_type_ids):
        if not isinstance(Phrase_type_ids, torch.Tensor):
            Phrase_type_ids = torch.tensor(
                Phrase_type_ids, dtype=torch.long, device=batch_Phrase_emb.device, requires_grad=False
            )
        # batch_Phrase_type_emb = []
        # for id_ in Phrase_type_ids:
        #     batch_Phrase_type_emb.append(self.phrase_attribute_emb_all[id_])
        # batch_Phrase_type_emb = torch.stack(batch_Phrase_type_emb, dim=0)
        batch_Phrase_type_emb = self.phrase_attribute_emb_all[Phrase_type_ids].to(device='cuda')
        out = batch_Phrase_emb + batch_Phrase_type_emb
        out.to(device='cuda')
        #out = self.dropout(out)
        return out

# config = AutoConfig.from_pretrained(
#         "bert-base-cased",
#     )

# type_encode = PhraseTypeEncoder(config, hidden_size=768)

# Core arguments
nsubj = 'A nominal subject (nsubj) is a nominal which is the syntactic subject and the proto-agent of a clause. That is, it is in the position that passes typical grammatical test for subjecthood, and this argument is the more agentive, the do-er, or the proto-agent of the clause. This nominal may be headed by a noun, or it may be a pronoun or relative pronoun or, in ellipsis contexts, other things such as an adjective.'
nsubj = ''
obj = 'The object of a verb is the second most core argument of a verb after the subject. Typically, it is the noun phrase that denotes the entity acted upon or which undergoes a change of state or motion (the proto-patient).'
iobj = 'The indirect object of a verb is any nominal phrase that is a core argument of the verb but is not its subject or (direct) object.'
csubj = 'A clausal subject is a clausal syntactic subject of a clause, i.e., the subject is itself a clause. The governor of this relation might not always be a verb: when the verb is a copular verb, the root of the clause is the complement of the copular verb. The dependent is the main lexical verb or other predicate of the subject clause.'

# Non-core dependents
obl = 'The obl relation is used for a nominal (noun, pronoun, noun phrase) functioning as a non-core (oblique) argument or adjunct. This means that it functionally corresponds to an adverbial attaching to a verb, adjective or other adverb.s'
obl_loc = ''
obl_tmod = ''
obl_npmod = ''
dislocated = 'The dislocated relation is used for fronted or postposed elements that do not fulfill the usual core grammatical relations of a sentence. These elements often appear to be in the periphery of the sentence, and may be separated off with a comma intonation.'
advcl = 'An adverbial clause modifier is a clause which modifies a verb or other predicate (adjective, etc.), as a modifier not as a core complement. This includes things such as a temporal clause, consequence, conditional clause, purpose clause, etc. The dependent must be clausal (or else it is an advmod) and the dependent is the main predicate of the clause.'
advmod = 'An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase that serves to modify a predicate or a modifier word.'

# Nominal dependents
appos = 'An appositional modifier of a noun is a nominal immediately following the first noun that serves to define, modify, name, or describe that noun. It includes parenthesized examples, as well as defining abbreviations in one of these structures.appos is intended to be used between two nominals. In general, modulo punctuation, the two halves of an apposition can be switched. '
acl = 'acl stands for finite and non-finite clauses that modify a nominal. The head of the acl relation is the noun that is modified, and the dependent is the head of the clause that modifies the noun.'
acl_relcl = ''

# Coordination
conj = 'A conjunct is the relation between two elements connected by a coordinating conjunction, such as and, or, etc.'

# Loose
list = 'The list relation is used for chains of comparable items. '
parataxis = 'The parataxis relation is a relation between a word (often the main predicate of a sentence) and other elements, such as a sentential parenthetical or a clause after a “:” or a “;”, placed side by side without any explicit coordination, subordination, or argument relation with the head word. Parataxis is a discourse-like equivalent of coordination, and so usually obeys an iconic ordering. '







