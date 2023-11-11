import numpy as np
import os

# Select dataset through task_id
task = ['UFFT', 'MA']
task_id = 0
print(task[task_id])

if task_id == 0:
    # UFFT
    all_sub = 30
elif task_id == 1:
    # MA
    all_sub = 29


all_acc = []
all_pre = []
all_rec = []
all_f1 = []
all_kap = []
for n_sub in range(1, all_sub + 1):
    sub_acc = []
    sub_pre = []
    sub_rec = []
    sub_f1 = []
    sub_kap = []
    for tr in range(5):
        path = os.path.join('save', task[task_id], 'KFold', str(n_sub), str(tr+1))
        val_acc = open(path + '/metrics.txt', "r")
        string = val_acc.readlines()[-1]

        acc = string.split('acc=')[1].split(', pre')[0]
        pre = string.split('pre=')[1].split(', rec')[0]
        rec = string.split('rec=')[1].split(', f1')[0]
        f1 = string.split('f1=')[1].split(', kap')[0]
        kappa = string.split('kap=')[1]

        acc = float(acc)
        pre = float(pre)
        rec = float(rec)
        f1 = float(f1) * 100
        kappa = float(kappa)

        sub_acc.append(acc)
        sub_pre.append(pre)
        sub_rec.append(rec)
        sub_f1.append(f1)
        sub_kap.append(kappa)

        all_acc.append(acc)
        all_pre.append(pre)
        all_rec.append(rec)
        all_f1.append(f1)
        all_kap.append(kappa)

    sub_acc = np.array(sub_acc)
    sub_pre = np.array(sub_pre)
    sub_rec = np.array(sub_rec)
    sub_f1 = np.array(sub_f1)
    sub_kap = np.array(sub_kap)
    print('\nsub = %d : acc = %.2f ± %.2f' % (n_sub, np.mean(sub_acc), np.std(sub_acc)))
    print('sub = %d : pre = %.2f ± %.2f' % (n_sub, np.mean(sub_pre), np.std(sub_pre)))
    print('sub = %d : rec = %.2f ± %.2f' % (n_sub, np.mean(sub_rec), np.std(sub_rec)))
    print('sub = %d :  f1 = %.2f ± %.2f' % (n_sub, np.mean(sub_f1), np.std(sub_f1)))
    print('sub = %d : kap = %.2f ± %.2f' % (n_sub, np.mean(sub_kap), np.std(sub_kap)))


print('\n=======> KFold-CV results of all subjects on ' + task[task_id])
all_acc = np.array(all_acc)
all_pre = np.array(all_pre)
all_rec = np.array(all_rec)
all_f1 = np.array(all_f1)
all_kap = np.array(all_kap)
print('acc = %.2f ± %.2f' % (np.mean(all_acc), np.std(all_acc)))
print('pre = %.2f ± %.2f' % (np.mean(all_pre), np.std(all_pre)))
print('rec = %.2f ± %.2f' % (np.mean(all_rec), np.std(all_rec)))
print('f1  = %.2f ± %.2f' % (np.mean(all_f1), np.std(all_f1)))
print('kap = %.2f ± %.2f' % (np.mean(all_kap), np.std(all_kap)))
