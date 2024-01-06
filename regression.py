import json
import os

reg_file = 'reg.json'
alternative_reg_file = 'reg2.json'
WAIVERS = ['parsing', 'multiple_class_lm', 'error_in_caption']

class RegressionHandler:
    def __init__(self, default_method=True):
        if default_method:
            self.reg_file = reg_file
        else:
            self.reg_file = alternative_reg_file
        if os.path.isfile(self.reg_file):
            with open(self.reg_file, 'r') as fp:
                self.reg, self.waivers = json.load(fp)
        else:
            self.reg = []
            self.waivers = {}

    def save(self):
        with open(self.reg_file, 'w') as fp:
            fp.write(json.dumps([self.reg, self.waivers]))

    def append(self, sample):
        self.reg.append(sample)

    def waive(self, ind, reason):
        assert reason in WAIVERS
        self.waivers[ind] = reason

    def remove_waiver(self, ind):
        if str(ind) in self.waivers:
            del self.waivers[ind]
