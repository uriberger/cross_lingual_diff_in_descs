import json
import os

reg_file = 'reg.json'

class RegressionHandler:
    def __init__(self):
        if os.path.isfile(reg_file):
            with open(reg_file, 'r') as fp:
                self.reg = json.load(fp)
        else:
            self.reg = []

    def save(self):
        with open(reg_file, 'w') as fp:
            fp.write(json.dumps(self.reg))

    def append(self, sample):
        self.reg.append(sample)
