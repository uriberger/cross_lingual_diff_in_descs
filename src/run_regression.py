from find_synsets_in_captions import find_synsets
from regression import RegressionHandler
import time

def run_regression():
    reg_obj = RegressionHandler()
    failed = []
    waived_and_passed = []
    waived_and_failed = []
    print('Running regression', flush=True)
    t = time.time()
    for i in range(len(reg_obj.reg)):
        if i % 1000 == 0:
            print(f'\tStarting sample {i} out of {len(reg_obj.reg)}, time from prev {time.time() - t}', flush=True)
            t = time.time()
        sample, gt = reg_obj.reg[i]
        res = find_synsets(sample['caption'])
        pred = [x[3] for x in res if x[3] is not None]
        if sorted(gt) != sorted(pred):
            if str(i) in reg_obj.waivers:
                waived_and_failed.append(i)
            else:
                failed.append(i)
        elif str(i) in reg_obj.waivers:
            waived_and_passed.append(i)
    passed_count = len(reg_obj.reg) - len(failed) - len(waived_and_failed) - len(waived_and_passed)
    print('Finished regression, results:')
    print(f'{passed_count} succeeded, {len(failed)} failed, {len(waived_and_passed)} passed with waiver, {len(waived_and_failed)} failed with waiver')
    print('Fail list:')
    print(failed)
    print('Passed with waiver:')
    print(waived_and_passed)
    print('Failed with waiver:')
    print(waived_and_failed)
