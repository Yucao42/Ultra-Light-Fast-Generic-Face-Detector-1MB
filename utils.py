import time
import logging
import os

class Timer:
    def __init__(self, op):
        self.op = op

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        finish = time.time()
        print("Time spent for {}: {:.4f}s".format(self.op, finish - self.start))
