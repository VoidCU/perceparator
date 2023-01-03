from dataset.preprocess import preprocess_one_dir
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import json

import torch


class MyDataset(Dataset):
    def __init__(self, data_dir, batch_size, sample_rate, segment):

        super(MyDataset, self).__init__()

        self.data_dir = data_dir
        self.sr = sample_rate
        self.batch_size = batch_size
        self.segment = segment

        file = ["mix", "s1", "s2"]

        self.mix_dir = os.path.join(data_dir, file[0])
        self.mix_list = os.listdir(os.path.abspath(self.mix_dir))

        self.s1_dir = os.path.join(data_dir, file[1])
        self.s1_list = os.listdir(os.path.abspath(self.s1_dir))

        self.s2_dir = os.path.join(data_dir, file[2])
        self.s2_list = os.listdir(os.path.abspath(self.s2_dir))

    def __getitem__(self, item):

        mix_path = os.path.join(self.mix_dir, self.mix_list[item])
        mix_data = librosa.load(
            path=mix_path,
            sr=self.sr,
            mono=True,  # Single channel
            offset=0,  # Audio read the starting point
            duration=None,  # Get audio time
            dtype=np.float32,
            res_type="kaiser_best",
        )[0]
        length = len(mix_data)

        s1_path = os.path.join(self.s1_dir, self.s1_list[item])
        s1_data = librosa.load(
            path=s1_path,
            sr=self.sr,
            mono=True,  # 单通道
            offset=0,  # 音频读取起始点
            duration=None,  # 获取音频时长
        )[0]

        s2_path = os.path.join(self.s2_dir, self.s2_list[item])
        s2_data = librosa.load(
            path=s2_path,
            sr=self.sr,
            mono=True,  # 单通道
            offset=0,  # 音频读取起始点
            duration=None,  # 获取音频时长
        )[0]

        s_data = np.stack((s1_data, s2_data), axis=0)

        return mix_data, length, s_data

    def __len__(self):

        return len(self.mix_list)


class EvalDataset(Dataset):
    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):

        super(EvalDataset, self).__init__()

        assert mix_dir != None or mix_json != None

        if mix_dir is not None:
            preprocess_one_dir(mix_dir, mix_dir, "mix", sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, "mix.json")

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)

        def sort(infos):
            return sorted(infos, key=lambda info: int(info[1]), reverse=True)

        sorted_mix_infos = sort(mix_infos)

        mini_batch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            mini_batch.append([sorted_mix_infos[start:end], sample_rate])

            if end == len(sorted_mix_infos):
                break

            start = end

        self.minibatch = mini_batch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(DataLoader):
    """
    NOTE: just use batch_size = 1 here, so drop_last = True makes no sense here.
    """

    def __init__(self, *args, **kwargs):

        super(EvalDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(num_batch):
    """
    Args:
        num_batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens: B, torch.Tentor
        filenames: a list contain B strings
    """
    assert len(num_batch) == 1

    mixtures, filenames = load_mixtures(num_batch[0])

    ilens = np.array([mix.shape[0] for mix in mixtures])  # 获取输入序列长度的批处理

    pad_value = 0

    mixtures_pad = pad_list(
        [torch.from_numpy(mix).float() for mix in mixtures], pad_value
    )  # 填充 0

    ilens = torch.from_numpy(ilens)

    return mixtures_pad, ilens, filenames


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []

    mix_infos, sample_rate = batch

    for mix_info in mix_infos:
        mix_path = mix_info[0]

        mix, _ = librosa.load(mix_path, sr=sample_rate)

        mixtures.append(mix)
        filenames.append(mix_path)

    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    print(n_batch)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":

    dataset = MyDataset(
        data_dir="C:/Users/86188/Desktop/Speech_Separation/dataset/min/tr/", sr=8000
    )

    data_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True
    )

    for (i, data) in enumerate(data_loader):

        if i >= 1:
            break

        mix, length, s = data
        print(mix.shape, length, s.shape)
