from torch.utils.data import Dataset,DataLoader
from easydict import EasyDict
import json
import random
import cv2
import imutils
import itertools

fast_val_n=100

def exe_aug(img):
    aug_func=[distort, stretch, perspective, scale, compress]
    n_fun=random.randint(0, len(aug_func))
    fun2exe=random.sample(aug_func, n_fun)
    for fun in fun2exe:
        if fun==distort: img=distort(img,segment=random.randint(3, 10))# [3, 10]
        if fun==stretch: img=stretch(img,segment=random.randint(3, 10))# [3, 10]
        if fun==perspective: img=perspective(img)
        if fun==scale: img=scale(img,hs=random.uniform(0.8, 1),ws=random.uniform(0.8, 1),hoj=random.uniform(0.8, 1.2),woj=random.uniform(0.8, 1.2))# [1, 0.8] [1, 0.8] [1.2, 0.8] [1.2, 0.8]
        if fun==compress: img=compress(img,r=random.uniform(0.8, 1))# [1, 0.8]
    return img

class Uni_dataset(Dataset):
    def __init__(self, datasets_name, phase, scaled_heights=[128]):
        super(Uni_dataset, self).__init__()

        self.phase=phase

        print('going to load', datasets_name)
        self.datasets_name=datasets_name
        self.dataset_dic=self.instancize(datasets_name, scaled_heights)
        self.tasks=[]
        for dataset in self.dataset_dic.values():
            if dataset.task not in self.tasks:
                self.tasks.append(dataset.task)

    def instancize(self, datasets_name, scaled_heights):
        dataset_dic={}
        for dataset in datasets_name:
            if dataset=='hme100k':
                dataset_dic[dataset]=HME100K(phase=self.phase,scaled_heights=scaled_heights)
            elif dataset=='mlhme38k':
                dataset_dic[dataset]=MLHME38K(phase=self.phase)
            elif dataset=='m2e':
                dataset_dic[dataset]=M2E(phase=self.phase, scaled_heights=scaled_heights)
            # if dataset=='crohme':
            #     dataset_lists.append(Crohme(phase=self.phase if self.phase != 'test' else '2014'))
            else:
                print(f'unknow dataset={dataset}')
                raise NotImplementedError

        return dataset_dic

    def __getitem__(self, index):
        if self.phase=='train':
            this_task=random.choice(self.tasks)
            dataset_name=''
            if this_task=='math': dataset_name='m2e'
            
            payload=self.dataset_dic[dataset_name][0]

        if self.phase=='valid':
            dataset_index=index // fast_val_n
            inset_index=index % fast_val_n
            dataset_name=self.datasets_name[dataset_index]
            payload=self.dataset_dic[dataset_name][inset_index]
        
        if self.phase=='test':
            dataset_name=self.datasets_name[0]
            payload=self.dataset_dic[dataset_name][index]
            
        return  dataset_name, *payload

    def __len__(self):
        if self.phase=='train':
            # return 50
            return 180_000
        elif self.phase=='valid':
            return fast_val_n*len(self.datasets_name)
        elif self.phase=='test':
            return len(self.dataset_dic[self.datasets_name[0]])
        else:
            raise NotImplementedError

class M2E(Dataset):
    def __init__(self, phase, scaled_heights=128):
        super(M2E, self).__init__()
        self.task='math'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]
        self.scaled_heights=scaled_heights

        runset='train' if phase=='train' else 'test'
        with open(self.dataRoot+f'/M2E/m2e.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        # self.lists=data['train']
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)
        print(f'm2e list loaded')

    def calibrate_dic(self, meta):
        for i,c in enumerate(meta):
            if c==r'\varnothing': meta[i]=r'\phi'
            if c==r'\Phi': meta[i]=r'\phi'
            if c==r'\left\{': meta[i]=r'\{'
            if c==r'\right\}': meta[i]=r'\}'
            if c==r'\left[': meta[i]=r'['
            if c==r'\left]': meta[i]=r']'
            if c==r'|': meta[i]=r'\vert'
            if c==r'\mid': meta[i]=r'\vert'
            if c==r'<': meta[i]=r'\leq'
            if c==r'>': meta[i]=r'\geq'
            if c==r'\leqslant': meta[i]=r'\leq'
            if c==r'\geqslant': meta[i]=r'\geq'
            if c==r'*': meta[i]=r'\ast'
        return meta
    
    def random_scale(self, img):
        # heights=list(range(232,296+1,8))
        heights=list(range(192,336+1,8))
        # heights=list(range(152,376+1,8))
        # heights=list(range(192-40,336+1-40,8))
        height=random.choice(heights)
        img=imutils.resize(img,height=height)
        return img
    
    def all_scales(self, img):
        all_scale_imgs=[]
        for height in self.scaled_heights:
            img=imutils.resize(img.copy(),height=height)
            all_scale_imgs.append(img)
        return all_scale_imgs

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1) if self.phase=='train' else index
        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup=entity_dic['meta']
        meta=entity_dic['meta']
        # meta=self.calibrate_dic(meta)

        runset='train' if self.phase=='train' else 'test'
        img = cv2.imread(self.dataRoot+f'/M2E/images/{name}', cv2.IMREAD_COLOR)
        assert img is not None

        if self.phase!='test': 
            img=self.random_scale(img)
        else:
            # img=imutils.resize(img,height=self.scaled_height)
            all_scale_imgs = self.all_scales(img)
            return name, all_scale_imgs, markup, meta, self.task

        # if img.shape[0]>img.shape[1]:
        #     img=imutils.resize(img,height=256)
        # else:
        #     img=imutils.resize(img,width=256)

        # if self.phase=='train': img=exe_aug(img)
        # cv2.imwrite(f'temp/{name}-{random.uniform(0,1)}.jpg',img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class HME100K(Dataset):
    def __init__(self, phase, scaled_heights=128):
        super(HME100K, self).__init__()
        self.task='math'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]
        self.scaled_heights=scaled_heights

        runset='train' if phase=='train' else 'test'
        with open(self.dataRoot+f'/HME100K/hme100k.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        # self.lists=data['train']
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'hme100k list loaded')

    def calibrate_dic(self, meta):
        for i,c in enumerate(meta):
            if c==r'\varnothing': meta[i]=r'\phi'
            if c==r'\Phi': meta[i]=r'\phi'
            if c==r'\left\{': meta[i]=r'\{'
            if c==r'\right\}': meta[i]=r'\}'
            if c==r'\left[': meta[i]=r'['
            if c==r'\left]': meta[i]=r']'
            if c==r'|': meta[i]=r'\vert'
            if c==r'\mid': meta[i]=r'\vert'
            if c==r'<': meta[i]=r'\leq'
            if c==r'>': meta[i]=r'\geq'
            if c==r'\leqslant': meta[i]=r'\leq'
            if c==r'\geqslant': meta[i]=r'\geq'
            if c==r'*': meta[i]=r'\ast'
        return meta
    
    def random_scale(self, img):
        heights=list(range(64,224+1,8))
        height=random.choice(heights)
        img=imutils.resize(img,height=height)
        return img

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1) if self.phase=='train' else index
        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup=entity_dic['meta']
        meta=entity_dic['meta']
        meta=self.calibrate_dic(meta)

        runset='train' if self.phase=='train' else 'test'
        # runset='train'
        name='test_18238.jpg'
        img = cv2.imread(self.dataRoot+f'/HME100K/{runset}_images/{name}', cv2.IMREAD_COLOR)
        assert img is not None

        img=imutils.resize(img, height=self.scaled_heights[0])
        # img=self.random_scale(img)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.merge([gray_image, gray_image, gray_image])

        # if self.phase!='test': 
        #     img=self.random_scale(img)
        # else:
        #     img=imutils.resize(img,height=self.scaled_height)

        # if img.shape[0]>img.shape[1]:
        #     img=imutils.resize(img,height=256)
        # else:
        #     img=imutils.resize(img,width=256)

        # if self.phase=='train': img=exe_aug(img)
        # cv2.imwrite(f'temp/{name}-{random.uniform(0,1)}.jpg',img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class MLHME38K(Dataset):
    def __init__(self, phase):
        super(MLHME38K, self).__init__()
        self.task='math'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]

        runset=''
        if phase=='train': runset='train'
        if phase=='valid': runset='Aset'
        if phase=='test': runset='Bset'
        with open(self.dataRoot+f'/MLHME38K/mlhme38k.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'mlhme38k list loaded')

    def calibrate_dic(self, meta):
        for i,c in enumerate(meta):
            if c==r'\varnothing': meta[i]=r'\phi'
            if c==r'\Phi': meta[i]=r'\phi'
            if c==r'\left\{': meta[i]=r'\{'
            if c==r'\right\}': meta[i]=r'\}'
            if c==r'\left[': meta[i]=r'['
            if c==r'\left]': meta[i]=r']'
            if c==r'|': meta[i]=r'\vert'
            if c==r'\mid': meta[i]=r'\vert'
            if c==r'<': meta[i]=r'\leq'
            if c==r'>': meta[i]=r'\geq'
            if c==r'\leqslant': meta[i]=r'\leq'
            if c==r'\geqslant': meta[i]=r'\geq'
            if c==r'*': meta[i]=r'\ast'
        return meta

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup=entity_dic['meta']
        meta=entity_dic['meta']
        meta=self.calibrate_dic(meta)

        runset=''
        if self.phase=='train': runset='train'
        if self.phase=='valid': runset='Aset'
        if self.phase=='test': runset='Bset'
        img = cv2.imread(self.dataRoot+f'/MLHME38K/{runset}/{name}.jpg', cv2.IMREAD_COLOR)
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)
        # cv2.imwrite(f'tmp/{name}-{random.uniform(0,1)}.jpg',img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)
    
class CSDB_mini(Dataset):
    def __init__(self, phase):
        super(CSDB_mini, self).__init__()
        self.task='chemistry'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]

        runset='train' if self.phase=='train' else 'test'
        with open(self.dataRoot+f'/CSDB/Mini-CASIA-CSDB/CSDB_mini.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'csdb_mini list loaded')

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup='\n'.join(entity_dic['transcription'])
        meta=entity_dic['meta']
        runset='train' if self.phase=='train' else 'test'
        img = cv2.imread(self.dataRoot+f'/CSDB/Mini-CASIA-CSDB/{runset}_images/{name}.png', cv2.IMREAD_COLOR)
        
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class Zinc(Dataset):
    def __init__(self, phase):
        super(Zinc, self).__init__()
        self.task='chemistry'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]

        runset=phase
        with open(self.dataRoot+f'/ZINC/zinc.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]

        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'zinc list loaded')

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup='\n'.join(entity_dic['transcription'])
        meta=entity_dic['meta']

        img = cv2.imread(self.dataRoot+f'/ZINC/png/{name}.png', cv2.IMREAD_COLOR)
        
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class Table_bank(Dataset):
    def __init__(self, phase):
        super(Table_bank, self).__init__()
        self.task='table'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]

        runset=phase
        with open(self.dataRoot+f'/TableBank/table_bank.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'table_bank list loaded')

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup='\n'.join(entity_dic['transcription'])
        meta=entity_dic['meta']

        img = cv2.imread(self.dataRoot+f'/TableBank/images/{name}.png', cv2.IMREAD_COLOR)
        
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class Didi_text(Dataset):
    def __init__(self, phase):
        super(Didi_text, self).__init__()
        self.task='flowchart'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]

        runset=phase
        with open(self.dataRoot+f'/DIDI/didi.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'didi_text list loaded')

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup='\n'.join(entity_dic['transcription'])
        meta=entity_dic['meta']

        img = cv2.imread(self.dataRoot+f'/DIDI/vis_text/{name}.png', cv2.IMREAD_COLOR)
        
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class Didi_no_text(Dataset):
    def __init__(self, phase):
        super(Didi_no_text, self).__init__()
        self.task='flowchart'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]

        runset=phase
        with open(self.dataRoot+f'/DIDI/didi_no_text.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]
        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'didi_no_text list loaded')

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup='\n'.join(entity_dic['transcription'])
        meta=entity_dic['meta']

        img = cv2.imread(self.dataRoot+f'/DIDI/vis_no_text/{name}.png', cv2.IMREAD_COLOR)
        
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)

class Primus(Dataset):
    def __init__(self, phase):
        super(Primus, self).__init__()
        self.task='music'
        self.dataRoot='../datasets'
        self.phase=phase
        self.lists=[]
        self.task='music'
        runset='train' if phase=='train' else 'test'
        with open(self.dataRoot+f'/CameraPrIMuS/primus.json','r',encoding='utf8') as f:
            data = json.load(f)
        self.lists=data[runset]

        if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

        print(f'primus list loaded')

    def __getitem__(self, index):
        dataset_index=random.randint(0, len(self.lists)-1)

        entity_dic=self.lists[dataset_index]
        name=entity_dic['name']
        markup='\n'.join(entity_dic['transcription'])
        meta=entity_dic['meta']

        img = cv2.imread(self.dataRoot+f'/CameraPrIMuS/Corpus/{name}/{name}.png', cv2.IMREAD_COLOR)
        assert img is not None

        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=256)
        else:
            img=imutils.resize(img,width=256)

        if self.phase=='train': img=exe_aug(img)

        return name, img, markup, meta, self.task

    def __len__(self):
        return len(self.lists)
    
# class Crohme(Dataset):
#     def __init__(self, phase):
#         super(Crohme, self).__init__()

#         self.dataRoot='../datasets'
#         self.phase=phase
#         self.lists=[]

#         runset='train' if phase=='train' else '2014'
#         with open(self.dataRoot+f'/CROHME/crohme.json','r',encoding='utf8') as f:
#             data = json.load(f)
#         self.lists=data[runset]
#         if self.phase=='valid': self.lists=random.sample(self.lists, fast_val_n)

#         print(f'crohme list loaded')

#     def __getitem__(self, index):
#         dataset_index=random.randint(0, len(self.lists)-1)

#         entity_dic=self.lists[dataset_index]
#         name=entity_dic['name']
#         markup='\n'.join(entity_dic['transcription'])
#         meta=entity_dic['meta']

#         runset='train' if self.phase=='train' else '2014'
#         img = cv2.imread(self.dataRoot+f'/CROHME/{runset}/{name}.bmp', cv2.IMREAD_COLOR)
#         assert img is not None

#         if 2*img.shape[0]>img.shape[1]:
#             img=imutils.resize(img,height=128)
#         else:
#             img=imutils.resize(img,width=256)
#         if self.phase=='train':
#             img=exe_aug(img)
#         # cv2.imwrite(f'tmp/{name}-{random.uniform(0,1)}.jpg',img)

#         return name, img, markup, meta

    # def __len__(self):
    #     return len(self.lists)



if __name__=='__main__':
    datasets=['primus','crohme','didi','didi_no_text','table_bank','zinc']
    data=Uni_dataset(phase='train',datasets=datasets)
    
    from debug import visualize_tensor

    for i in range(24_000):
        print(data[i][0])
        visualize_tensor(data[i][1])
        print(data[i][2])
