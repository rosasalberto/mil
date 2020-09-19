from tensorflow.keras.utils import Progbar

class ProgressBar:
    """ tf keras progress bar to print the evaluations results """
    def __init__(self, metrics):
        self.metrics_train = ['train_' + metric for metric in metrics]
        self.metrics_val = ['val_' + metric for metric in metrics]
        self.metrics = self.metrics_train + self.metrics_val
        
    def prepare(self, n_samples, verbose=1):
        self.n_samples = n_samples
        self.prog_bar = Progbar(n_samples, stateful_metrics=self.metrics, 
                                verbose=verbose)
        
    def update_bar(self, d_train, d_val=[]):
        values = []
        for key in d_train: 
            values.append(('train_'+key, d_train[key]))
        for key in d_val:
            values.append(('val_'+key, d_val[key]))
        self.prog_bar.add(n=1, values=values)