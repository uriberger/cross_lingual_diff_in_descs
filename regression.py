import json
import os

reg_file = 'reg.json'
alternative_reg_file = 'reg2.json'
final_reg_file = 'reg3.json'
WAIVERS = ['parsing', 'multiple_class_lm', 'error_in_caption']

class RegressionHandler:
    def __init__(self, method='final'):
        if method == 'orig':
            self.reg_file = reg_file
        elif method == 'alternative':
            self.reg_file = alternative_reg_file
        elif method == 'final':
            self.reg_file = final_reg_file
        else:
            assert False, f'Uknown method {method}'
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
