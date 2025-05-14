import json
import pdb

all_dic={}
def up_dic(tex):
    for t in tex:
        if all_dic.get(t) is None:
            all_dic[t]=1
        else:
            all_dic[t]+=1

def file2list(phase):
    with open(f'{phase}_labels.txt','r',encoding='utf8')as f:
        data=f.readlines()

    the_list=[]
    for line in data:
        name,tex = line.strip().split('\t')
        tex=tex.split(' ')
        this_dic={'name':name, 'meta':tex}
        the_list.append(this_dic)
        up_dic(tex)
    return the_list


train_list=file2list('train')
test_list=file2list('test')

with open(f'hme100k.json','w',encoding='utf8')as f:
    dump_dic={'train':train_list, 'test':test_list}
    json.dump(dump_dic,f,indent='\t',ensure_ascii=False)

with open(f'dic_hme100k.txt','w',encoding='utf8')as f:
    for k in list(all_dic.keys()):
        f.write(f'{k}\n')

