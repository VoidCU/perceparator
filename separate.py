import os
import librosa
import torch
from dataset.data import DataLoader, MyDataset
from model.perceiver import Perceparator
from src.utils import remove_pad
import json5
import time


def main(config):

    if config["mix_dir"] is None and config["mix_json"] is None:
        print(
            "Must provide mix_dir or mix_json! When providing mix_dir, mix_json is ignored."
        )

    model = Perceparator.load_model(config["model_path"])

    model.eval()  # Set the model as the verification mode

    if torch.cuda.is_available():
        model.cuda()

    # Download Data
    eval_dataset = MyDataset(
        config["mix_dir"],
        config["mix_json"],
        batch_size=config["batch_size"],
        sample_rate=config["sample_rate"],
        # segment=config["segment"],
    )

    eval_loader = DataLoader(eval_dataset, batch_size=1)

    os.makedirs(config["out_dir"], exist_ok=True)
    os.makedirs(config["out_dir"] + "/mix/", exist_ok=True)
    os.makedirs(config["out_dir"] + "/s1/", exist_ok=True)
    os.makedirs(config["out_dir"] + "/s2/", exist_ok=True)

    # Audio generating function
    def write_wav(inputs, filename, sr=config["sample_rate"]):
        librosa.output.write_wav(filename, inputs, sr)  # norm=True)

    # Calculation without reverse transmission gradient
    with torch.no_grad():
        for (i, data) in enumerate(eval_loader):
            print("{}-th Batch Data Start Generate".format(i))

            start_time = time.time()
            mixture, mix_lengths, filenames = data

            # if torch.cuda.is_available():

            #     mixture = mixture.cuda()

            #     mix_lengths = mix_lengths.cuda()

            estimate_source = model(mixture)  # Put the data in the model

            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)

            mixture = remove_pad(mixture, mix_lengths)

            for i, filename in enumerate(filenames):
                filename = os.path.join(
                    config["out_dir"] + "/mix/",
                    os.path.basename(filename).strip(".wav"),
                )

                write_wav(mixture[i], filename + ".wav")

                C = flat_estimate[i].shape[0]

                for c in range(C):
                    if c == 0:
                        filename = os.path.join(
                            config["out_dir"] + "/s1/",
                            os.path.basename(filename).strip(".wav"),
                        )
                        write_wav(
                            flat_estimate[i][c], filename + "_s{}.wav".format(c + 1)
                        )
                    elif c == 1:
                        filename = os.path.join(
                            config["out_dir"] + "/s2/",
                            os.path.basename(filename).strip(".wav"),
                        )
                        write_wav(
                            flat_estimate[i][c], filename + "_s{}.wav".format(c + 1)
                        )
                    else:
                        print("Continue To Add")

            end_time = time.time()

            run_time = end_time - start_time

            print("Elapsed Time: {} s".format(run_time))

        print("Data Generation Completed")


if __name__ == "__main__":

    path = r"config/test/separate.json5"
    # args = parser.parse_args()
    with open(path) as f:
        configuration = json5.load(f)

    main(configuration)
