import json
import pdb

sub_sets=['train','val','test']

dic={}
meta={'train':[], 'val':[], 'test':[]}

def add_dic(tex, dic):
    tex = tex.split(' ')
    for token in tex:
        if dic.get(token, None) is None:
            dic[token]=1
        else:
            dic[token]+=1

for sub_set in sub_sets:
    with open(f'datasets/M2E/{sub_set}.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            tex=json_obj['tex'].strip()
            add_dic(tex, dic)
            entity_dic={'name':json_obj['name'],'meta':tex.split(' ')}
            meta[sub_set].append(entity_dic)

print(dic)
print(len(dic))

with open('datasets/M2E/dic_m2e.txt','w',encoding='utf8')as f:
    for k in list(dic.keys()):
        f.write(f'{k}\n')

with open('datasets/M2E/m2e.json','w',encoding='utf8')as f:
    json.dump(meta,f,indent='\t',ensure_ascii=False)
