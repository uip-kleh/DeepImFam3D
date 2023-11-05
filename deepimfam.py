import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from aaindex import aaindex1
import statistics
import numpy as np
from setconfig import SetConfig

class DeepImFam(SetConfig):
    def __init__(self) -> None:
        super().__init__()

    def load_GPCR(self):
        with open(self.gpcrTrainPath) as f:
            for l in f.readlines():
                subsubfam, aaseq = l.split()

if __name__ == '__main__':
    deepimfam = DeepImFam()
    deepimfam.define_aavector()
