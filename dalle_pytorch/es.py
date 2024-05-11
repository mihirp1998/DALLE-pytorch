import ipdb
st = ipdb.set_trace
import numpy as np



class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """
    Early stopping utility
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self, patience, mode="max", delta=0.001):
        # self.accelerator = accelerator
        # st()
        self.patience = int(patience)
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf


    def __call__(self, epoch_score):
        if self.early_stop:
            return False


        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)


        if self.best_score is None:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.best_score, epoch_score))
            self.best_score = score
            # self.save_checkpoint(epoch_score, model, model_path)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.best_score, epoch_score))
            self.best_score = score
            # self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0
            return True

    # def save_checkpoint(self, epoch_score, model, model_path):
    #     if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
    #         print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))


    #         self.callbac(self.accelerator.unwrap_model(model).state_dict(), model_path)
    #     self.val_score = epoch_score