import os
import librosa
import soundfile as sf
import torch
from model.perceiver import Perceparator


def main(path_of_data):
    model = Perceparator.load_model("./checkpoint/epoch245.pth.tar")
    model.eval()
    data, _ = sf.read(path_of_data)
    data = data[::2]
    # data = torch.from_numpy(data)
    data = torch.tensor(data).type(torch.Tensor).unsqueeze(0)
    print(data[0])
    estimate_source = model(data)
    estimate_source = estimate_source.squeeze(0)
    s1 = estimate_source[0]
    s2 = estimate_source[1]
    sf.write("./examinedata/N6_1.wav", s1.tolist(), 8000)
    sf.write("./examinedata/n6_2.wav", s2.tolist(), 8000)
    print(estimate_source.size())


if __name__ == "__main__":
    path_of_data = "./examinedata/6.wav"
    main(path_of_data)
