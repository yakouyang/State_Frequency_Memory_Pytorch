import argparse
import os

class OptionParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
        
    def set_arguments(self):
        # training setting 
        self.parser.add_argument('--filename', type=str, default="./Data/data.npy", help='datasets')
        self.parser.add_argument('--step', type=int, default=5, help='forward predict steps')
        self.parser.add_argument('--epochs', type=int, default=10, help='training epochs')
        self.parser.add_argument('--net', type=str, default="sfm", help='choose which net to use')
        self.parser.add_argument('--train', type=int, default=0, help='train')
        self.parser.add_argument('--test', type=int, default=1, help='test')
        
        # exchange variable
        self.parser.add_argument('--max_data', type=None, default=None, help='store max_data')
        self.parser.add_argument('--min_data', type=None, default=None, help='store min_data')
        self.parser.add_argument('--train_len', type=None, default=None, help='store train_len')
        self.parser.add_argument('--val_len', type=None, default=None, help='store val_len')
        self.parser.add_argument('--test_len', type=None, default=None, help='store test_len')
    
    def parse_args(self):
        opt = self.parser.parse_args()
        args = vars(opt)
        return opt

