import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from streaming import MDSWriter, StreamingDataset
import torch.distributed as dist
import numpy as np

from composer import Trainer
from composer.models import ComposerModel
from composer.utils import dist


class Model(nn.Module):
    """Toy convolutional neural network architecture in pytorch for MNIST."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes
        self.fc1 = nn.Linear(2, num_classes)

    def forward(self, x):
        out = self.fc1(x['feature'])
        time.sleep(0.2)
        return out


class MyComposerModel(ComposerModel):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = Model(num_classes)
        self.data = []

    def forward(self, batch):
        self.data += [batch['class'].cpu().numpy()]
        return self.model(batch)

    def loss(self, outputs, batch):
        loss = F.cross_entropy(outputs, batch['class']-1)
        return loss


def make_data():
    # Directory in which to store the compressed output files
    data_dir = '../tmp_mds/'

    # A dictionary mapping input fields to their data types
    columns = {
        'feature': 'ndarray',
        'class': 'int'
    }

    # Save the samples as shards using MDSWriter
    with MDSWriter(out=data_dir, columns=columns) as out:
        for i in range(32):
            sample = {
                'feature': np.array([i + 1, i + 1]).astype(np.float32),
                'class': i + 1,
            }
            print(sample)
            out.write(sample)


def composer_train_loop():
    # CUDA_VISIBLE_DEVICES=0,1 composer -v --stdout ./stdout_{rank}.txt --stderr ./stderr_{rank}.txt tmp.py
    local = './tmp_mds'

    device_train_microbatch_size = 1
    global_train_batch_size = 4
    device_batch_size = global_train_batch_size // dist.get_world_size()

    # Create streaming dataset
    # 32 samples in dataset
    dataset = StreamingDataset(local=local, remote=None, shuffle=False, shuffle_seed=42,
                               batch_size=device_batch_size,
                               shuffle_block_size=32,
                               predownload=32,
                               num_canonical_nodes=1)
    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=device_batch_size)

    print(f"[{dist.get_global_rank()}]: global_train_batch_size: {global_train_batch_size}")
    print(f"[{dist.get_global_rank()}]: device_train_microbatch_size: {device_train_microbatch_size}")
    print(f"[{dist.get_global_rank()}]: device_batch_size: {device_batch_size}")
    print(f"[{dist.get_global_rank()}]: len(dataset): {len(dataset)}")
    print(f"[{dist.get_global_rank()}]: dataset.size: {dataset.size}")

    model = MyComposerModel(num_classes=dataset.size)
    print(model)

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        device_train_microbatch_size=device_train_microbatch_size,
        max_duration="2ep",
        save_interval='1ep',
        save_folder='./runs/debug/tmp',
        save_num_checkpoints_to_keep=-1,
        # load_path='./runs/debug/tmp/ep1-ba8-rank0.pt',
        )
    trainer.fit()

    model_data = list(np.concatenate(model.data))
    print(f'[{dist.get_global_rank()}]: model.data: {model_data}')
    print(f'[{dist.get_global_rank()}]: Done')


def data_loop():
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tmp.py
    dist.init_process_group(backend='nccl')
    # Local working dir where dataset is cached during operation
    local = './tmp_mds'

    # Create streaming dataset
    dataset = StreamingDataset(local=local, remote=None, shuffle=False, shuffle_seed=42,
                               batch_size=1,
                               num_canonical_nodes=1)

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=1)

    rank = dist.get_rank()
    print(f"[{rank}]: len(dataset): {len(dataset)}")
    print(f"[{rank}]: dataset.size: {dataset.size}")

    data = []
    for idx, i in enumerate(dataloader):
        # if rank in [0, 1]:
        # import pdb;pdb.set_trace()
        # print(f"Rank: {rank}, data: {i}, idx: {idx}")
        data += [i['class'].item()]
    print(f"[{rank}]: data: {data}")


if __name__ == "__main__":
    # make_data()
    # data_loop()
    composer_train_loop()
