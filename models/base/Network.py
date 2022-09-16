import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from timm import create_model


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100', 'cifar100_1']:
            self.encoder = resnet20()
            self.num_features = 64
            # self.encoder = create_model('efficientnetv2_s', pretrained=True, features_only=False)
            # self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-2])
            # self.num_features = 1280
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            # pretrained=True follow TOPIC,
            # models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 平均池化

        # 增量学习中更新的是fc部分的权重
        # self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)  # 全连接
        self.fc = nn.Linear(self.num_features, self.args.base_class, bias=False)  # 创建基于基础类别的全连接

    def forward_metric(self, x):
        x = self.encode(x)  # 通过resnet
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            # using linear classifier
            x = self.fc(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            """Tensor.detach()
            
            将tensor从计算图中脱离出来
            假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B，那么可以这样做：
            input_B = output_A.detach()
            简言之，就是将模型A的参数冻结，下列代码将encode冻结
            """
            data = self.encode(data).detach()

        if self.args.not_data_init:  # False
            # new_fc: shape = [len(class_list), num_features)
            # cifar: self.num_features = 64, cub200&mini_imagenet = 512
            new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"), requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc, data, label, session)

    @torch.no_grad()
    def update_fc_avg(self, data, label, class_list):
        # data_init
        # data.shape = [25, 64], label.shape = [25,], class_list.shape = [5, ]

        # update self.fc.out_features
        fc = nn.Linear(self.num_features, self.fc.out_features + len(class_list), bias=False)
        fc.weight[:self.fc.out_features, :] = self.fc.weight
        fc.training = self.fc.training
        fc.to(self.fc.weight.device)
        self.fc = fc

        new_fc = []
        for class_index in class_list:
            # .nonzero()输出label==class_index时的索引
            # 对于5-way, 5-shot, data_index.shape = [5,]
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]  # [5, 64]
            proto = embedding.mean(0)  # [64,]
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self, new_fc, data, label, session):
        """update fully connect finetune

        Args:
            new_fc: new fully connect weights [5, num_features]
            data: output from resnet  [5, num_features]
            label: [5, ]
            session(int):
        """
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new,
                                    momentum=0.9, dampening=0.9, weight_decay=0)

        label = label.type(torch.long).to(self.args.device)
        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
