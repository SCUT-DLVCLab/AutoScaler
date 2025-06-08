from Levenshtein import distance

def deal_a_file(content):
    print(len(content))
    n_line=len(content)
    assert n_line%3==0
    acc=0
    n=n_line//3
    for i in range(n):
        name=content[i*3]
        pr=content[i*3+1]
        gt=content[i*3+2]

        name=name.replace('-','').strip()
        pr=pr[3:].strip()
        # pr=pr.split(' ')
        # pr=list(map(int,pr))

        gt=gt[3:].strip()
        # gt=gt.split(' ')
        # gt=list(map(int,gt))


        # print(pr)
        # print(gt)
        if pr==gt:
            acc+=1
    
    return acc/n

def read_a_file(epoch_id,scale):
    ans_path=f'ans/{epoch_id}_{scale}.txt'
    with open(ans_path,'r',encoding='utf8')as f:
        content=f.readlines()[:-1]
    n_line=len(content)
    assert n_line%3==0
    n=n_line//3
    data={}
    for i in range(n):
        name=content[i*3]
        pr=content[i*3+1]
        gt=content[i*3+2]

        name=name.replace('-','').strip()
        pr=pr[3:].strip()
        pr=pr.split(' ')
        if pr==['']:
            pr=[]
        else:
            pr=list(map(int,pr))

        gt=gt[3:].strip()
        gt=gt.split(' ')
        gt=list(map(int,gt))

        data[name]=[pr,gt]
    
    return data

def find_best(epoch_id):
    scales=list(range(32,129,8))
    datas={}
    for scale in scales:
        datas[scale]=read_a_file(epoch_id,scale)
    acc=0
    best_scale={}

    for name in datas[128].keys():
        for scale in scales:
            print(name)
            try:
                pr,gt=datas[scale][name]
                if pr==gt:
                    acc+=1
                    best_scale[name]=scale
                    break
            except KeyError as e:
                print(e)
                print(f'No data at {scale} {name}')
    # print(best_scale)
    print(acc/len(datas[32].keys()))
    print(len(datas[32].keys()))

find_best(199)