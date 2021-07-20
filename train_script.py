import os

lrs = [3e-5, 5e-5, 1e-5, 3e-5, 5e-5]
epochs = [15]*5
tasks = ['mrpc', 'qnli', 'qqp', 'sst2', 'rte']

for lr, epoch, task in zip(lrs, epochs, tasks):
    # Must have done accelerate config setting before you run this script
    os.system(f'accelerate launch train.py --task {task} --num_epochs {epoch} --learning_rate {lr}')
 