def parse_list(x):
    return [int(i.strip()) for i in x.split(',')]
import os
class MainConfig:

    def __init__(self, main_config, arg):
        self.batch_size = arg.batch_size
        self.eval_batch_size = int(main_config['TRAINING']['eval_batch_size'])
        self.dropout = float(main_config['TRAINING']['dropout'])

        self.eval_every = int(main_config['TRAINING']['eval_every'])
        self.checkpoints_to_keep = int(main_config['TRAINING']['checkpoints_to_keep'])

        self.save_every = int(main_config['TRAINING']['save_every'])
        self.patience = int(main_config['TRAINING']['patience'])
        self.log_device_placement = bool(main_config['TRAINING']['log_device_placement'])

        self.logs_path = str(main_config['DATA']['logs_path'])
        self.max_sequence_len = int(main_config['DATA']['max_sequence_len'])
        self.vocab_file = str(main_config['DATA']['vocab_file'])
        self.embeddings = os.path.join(arg.data_dir, main_config['DATA']['embeddings'])
        self.train_file = str(main_config['DATA']['train_file'])
        self.test_file = str(main_config['DATA']['test_file'])
        self.dev_file = str(main_config['DATA']['dev_file'])
        self.train_bpe_file = str(main_config['DATA']['train_bpe_file'])
        self.test_bpe_file = str(main_config['DATA']['test_bpe_file'])
        self.dev_bpe_file = str(main_config['DATA']['dev_bpe_file'])

        self.embedding_size = int(main_config['PARAMS']['embedding_size'])
        self.emb_vocab_size = int(main_config['PARAMS']['emb_vocab_size'])
        self.data_dir = arg.data_dir
        self.model_dir = arg.model_dir

def get_args(args):

    return args
