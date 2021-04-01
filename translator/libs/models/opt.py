import torch
import pickle

class opt(object):
    def __init__(self):
        # File Parameter
        self.dir = "translator/libs/models/"
        self.novel_path = None

        # Model hyperparameter
        # vocab size must be fixed to facebook fast text
        # mode is used for data_loader - train / validate
        self.mode = None
        self.ratio = 0.9
        self.n_layers = 2
        self.hidden_size = 2048
        self.embedding_size = 2048
        
        # Training hyperparameter
        self.cuda = True if torch.cuda.is_available() else False
        self.batch_size = 1
        self.seq_len = 50
        self.num_epochs = 100
        self.save_every = 50
        self.print_every = 10
        self.valid_every = 20
        self.grad_clip = 5
        self.lr = 0.01
        
        # Vocabulary setting - This will be filled automatically
        with open(self.dir + 'gru_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        self.vocab_size = vocab['vocab_size']
        self.vocab_itoc = vocab['vocab_itoc']
        self.vocab_ctoi = vocab['vocab_ctoi']

        # Check if we can use train / valid data
        self.train = True
        self.valid = True