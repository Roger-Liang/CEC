import torch
import numpy as np
import copy


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        # 30000 / 50 = 600
        self.n_batch = n_batch  # the number of iterations in the dataloader  50
        self.n_cls = n_cls  # 15  选取15个类别
        self.n_per = n_per  # 16
        # 15 * 16 = 240
        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class(60个类别的索引)
        for i in range(max(label) + 1):  # range(59 + 1)
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            # 从60个类别索引里随机选择15个类别索引
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexes of this class(l = 500)
                # 从500个数据中随机选择16个数据
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)  # batch就是15个类别中挑选16个数据得到
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
