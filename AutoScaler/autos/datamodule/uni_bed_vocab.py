import os
from easydict import EasyDict
from typing import Dict, List

class MasterEnVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    OOV_IDX = 3

    def __init__(self, tasks) -> None:
        self.tasks=tasks
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX
        self.task_vocabs=EasyDict()

        print(f'Loading {tasks} tasks')

        for task in tasks:
            auto_path='../datasets/M2E/dic_m2e.txt'
            if not os.path.exists(auto_path):
                auto_path='../'+auto_path
            with open(auto_path,'r',encoding='utf8')as f:
                for line in f.readlines():
                    w = line.strip()
                    if w=='': w=' '
                    w=f'{task[:2]}:'+w
                    self.word2idx[w] = len(self.word2idx)

            self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        print(f"Init envocab with size: {len(self.word2idx)}")

    def words2indices(self, words: List[str],task) -> List[int]:
        ans=[]
        for w in words:
            try:
                ans.append(self.word2idx[task[:2]+':'+w])
            except KeyError:
                # ans.append()
                print(w)
                import pdb; pdb.set_trace()
        return ans

    def indices2words(self, id_list: List[int], task_tag=True) -> List[str]:
        words_list=[]
        for i in id_list:
            w=self.idx2word[i]
            if not task_tag: w=w[3:]
            words_list.append(w)
        return words_list

    def indices2label(self, id_list: List[int], task_tag=True) -> str:
        words = self.indices2words(id_list, task_tag)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

class MasterDeVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    OOV_IDX = 3

    def __init__(self, tasks) -> None:
        self.tasks=tasks
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX
        self.task_vocabs=EasyDict()

        print(f'Loading {tasks} tasks')
        for task in self.tasks:
            task_vocab=SubVocab(task)
            # task_vocab=globals()[class_name]()
            self.task_vocabs[task]=task_vocab

        # print(f"Init vocab with size: {len(self.word2idx)}")

    def words2indices(self, task, words: List[str]) -> List[int]:
        return self.task_vocabs.task.words2indices(words)
        
    def indices2words(self, task, id_list: List[int]) -> List[str]:
        return self.task_vocabs.task.indices2words(id_list)

    def indices2label(self, task, id_list: List[int]) -> str:
        return self.task_vocabs.task.indices2label(id_list)

    def __len__(self):
        return sum([len(self.task_vocabs.task) for task in self.tasks])

class SubVocab:
    def __init__(self,task) -> None:
        self.word2idx = dict()
        self.task=task
        auto_path='../datasets/M2E/dic_m2e.txt'
        with open(auto_path,'r',encoding='utf8')as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        # print(f"Init vocab with size: {len(self.word2idx)}")

    def words2indices(self, words: List[str]) -> List[int]:
        ans=[]
        for w in words:
            try:
                ans.append(self.word2idx[w])
            except KeyError:
                print(w)
                import pdb; pdb.set_trace()
        return ans

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)