import json
import os

reg_file = 'reg.json'
WAIVERS = ['parsing', 'multiple_class_lm', 'error_in_caption']

class RegressionHandler:
    def __init__(self):
        if os.path.isfile(reg_file):
            with open(reg_file, 'r') as fp:
                self.reg, self.waivers = json.load(fp)
        else:
            self.reg = []
            self.waivers = {}

    def save(self):
        with open(reg_file, 'w') as fp:
            fp.write(json.dumps([self.reg, self.waivers]))

    def append(self, sample):
        self.reg.append(sample)

    def waive(self, ind, reason):
        assert reason in WAIVERS
        self.waivers[ind] = reason

    def remove_waiver(self, ind):
        if str(ind) in self.waivers:
            del self.waivers[ind]
