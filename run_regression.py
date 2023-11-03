from find_classes_in_caption import find_classes
from regression import RegressionHandler
import time

def run_regression():
    reg_obj = RegressionHandler()
    failed = []
    print('Running regression', flush=True)
    t = time.time()
    for i in range(len(reg_obj.reg)):
        if i % 1000 == 0:
            print(f'\tStarting sample {i} out of {len(reg_obj.reg)}, time from prev {time.time() - t}', flush=True)
            t = time.time()
        sample, gt = reg_obj.reg[i]
        res = find_classes(sample['caption'])
        pred = set([x[2] for x in res if x is not None])
        if gt != pred:
            failed.append(i)
    print('Finished regression, results:')
    print(f'{len(reg_obj.reg) - len(failed)} succeeded, {len(failed)} failed. Fail list:')
    print(failed)
