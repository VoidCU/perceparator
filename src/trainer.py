import os
import time
import torch
from src.pit_criterion import cal_loss_pit, cal_loss_no, MixerMSE
from torch.utils.tensorboard import SummaryWriter
import gc


class Trainer(object):
    def __init__(self, data, model, optimizer, config):

        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = config["train"]["use_cuda"]  # use or not GPU
        self.epochs = config["train"]["epochs"]  # Training batches
        #WHETHER to Adjust The Learning Rate
        self.half_lr = config["train"]["half_lr"]
        self.early_stop = config["train"]["early_stop"]  # Whether to stop early
        self.max_norm = config["train"]["max_norm"]  # L2 Sample

        # save and load model
        self.save_folder = config["save_load"]["save_folder"]  # Model preservation path
        self.checkpoint = config["save_load"]["checkpoint"]  # Whether to save each training model
        # Whether the original training progress is carried out
        self.continue_from = config["save_load"]["continue_from"]
        self.model_path = config["save_load"]["model_path"]  # Model preservation format

        # logging
        self.print_freq = config["logging"]["print_freq"]

        # loss
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        # File folder to generate a model
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_improve = 0

        # Visualization
        self.write = SummaryWriter("./logs")

        self._reset()

        self.MixerMSE = MixerMSE()

    def _reset(self):
        if self.continue_from:
            # Then the original progress training
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.save_folder + self.continue_from)

            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module

            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])

            self.start_epoch = int(package.get('epoch', 1))

            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            # Re -training
            self.start_epoch = 0

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
#             print("Train Start...")

#             self.model.train()  # Set the model to training mode

#             start_time = time.time()  # Training start time

#             tr_loss = self._run_one_epoch(epoch)  # Training model

#             gc.collect()
#             torch.cuda.empty_cache()

#             self.write.add_scalar("train loss", tr_loss, epoch+1)

#             end_time = time.time()  # Time of training
#             run_time = end_time - start_time  # Training time

#             print('-' * 85)
#             print('End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(
#                 epoch+1, run_time, tr_loss))
#             print('-' * 85)

#             if self.checkpoint:
#                 # Save each training model
#                 file_path = os.path.join(
#                     self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))

#                 if self.continue_from == "":
#                     if isinstance(self.model, torch.nn.DataParallel):
#                         self.model = self.model.module

#                 torch.save(self.model.serialize(self.model,
#                                                 self.optimizer,
#                                                 epoch + 1,
#                                                 tr_loss=self.tr_loss,
#                                                 cv_loss=self.cv_loss), file_path)

#                 print('Saving checkpoint model to %s' % file_path)

            print('Cross validation Start...')
            gc.collect()
            torch.cuda.empty_cache()
            self.model.eval()  # Set the model as the verification mode

            start_time = time.time()  # Verification start time

            val_loss = self._run_one_epoch(
                epoch, cross_valid=True)  # Verification model

            self.write.add_scalar("validation loss", val_loss, epoch+1)

            end_time = time.time()  # Verification end time
            run_time = end_time - start_time  # Training time

            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | ''Valid Loss {2:.3f}'.format(
                epoch+1, run_time, val_loss))
            print('-' * 85)

            # Whether to adjust the learning rate
            if self.half_lr:
                # Verify whether the loss is increased
                if val_loss >= self.prev_val_loss:
                    self.val_no_improve += 1  # The number of statistics has not been improved

                    # If the training 3 EPOCH has not been improved, the learning rate is halved
                    if self.val_no_improve >= 3:
                        self.halving = True

                    # If the 10 EPOCH has not been improved, the training is over
                    if self.val_no_improve >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_improve = 0

            if self.halving:
                optime_state = self.optimizer.state_dict()
                optime_state['param_groups'][0]['lr'] = optime_state['param_groups'][0]['lr']/2.0
                self.optimizer.load_state_dict(optime_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optime_state['param_groups'][0]['lr']))
                self.halving = False

            self.prev_val_loss = val_loss  # Current loss

            self.tr_loss[epoch] = tr_loss
            self.cv_loss[epoch] = val_loss

           # Save the best model
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss  # Minimum verification loss value

                file_path = os.path.join(self.save_folder, self.model_path)

                torch.save(self.model.serialize(self.model,
                                                self.optimizer,
                                                epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss), file_path)

                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):

        start_time = time.time()

        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader  # Data set switch

        for i, (data) in enumerate(data_loader.dataset):

            padded_mixture, mixture_lengths, padded_source = data
            # print(type(mixture_lengths))
            # print(mixture_lengths)
            padded_mixture = torch.tensor(padded_mixture, dtype=torch.float32)
            mixture_lengths = torch.tensor(
                mixture_lengths, dtype=torch.float32)
            mixture_lengths = mixture_lengths.unsqueeze(0)
            padded_source = torch.tensor(padded_source, dtype=torch.float32)
            gc.collect()
            torch.cuda.empty_cache()
            # use or not GPU train
            # if torch.cuda.is_available():
            #     padded_mixture = padded_mixture.cuda()
            #     mixture_lengths = mixture_lengths.cuda()
            #     padded_source = padded_source.cuda()

            # milking = padded_mixture.cuda()
            # estimate_source = self.model(milking)
            # print(torch.cuda.max_memory_allocated())

            estimate_source = self.model(padded_mixture)  # Put the data in the model
            loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_pit(padded_source,
                                                                                   estimate_source,
                                                                                   mixture_lengths)

            # loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_no(padded_source,
            #                                                                       estimate_source,
            #                                                                       mixture_lengths)

            # loss = self.MixerMSE(estimate_source, padded_source)

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            end_time = time.time()
            run_time = end_time - start_time

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Current Loss {3:.6f} | {4:.1f} s/batch'.format(
                    epoch+1,
                    i+1,
                    total_loss/(i+1),
                    loss.item(),
                    run_time/(i+1)),
                    flush=True)

        return total_loss/(i+1)
