import csv
import os
import matplotlib.pyplot as plt

if os.path.exists('metrics.csv'):
    path='metrics.csv'
else:
    v=input('input version:\n')
    path=f'./lightning_logs/version_{v}/metrics.csv'

datastes=['primus','crohme','didi','didi_no_text','table_bank','zinc']

with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    cur_lr=float('Inf')
    cur_step=0
    cur_loss=float('Inf')
    cur_cer=float('Inf')
    cur_epoch=0
    cur_vloss=float('Inf')
    cur_tloss=float('Inf')
    mean_cer=float('Inf')

    primus_cer=float('Inf')
    crohme_cer=float('Inf')
    didi_cer=float('Inf')
    didi_no_text_cer=float('Inf')
    table_bank_cer=float('Inf')
    zinc_cer=float('Inf')

    primus_exp=0
    crohme_exp=0
    didi_exp=0
    didi_no_text_exp=0
    table_bank_exp=0
    zinc_exp=0

    needed_data=[]
    lr_change=[]

    for row in reader:
        if 'step' in row and row['step']!='':
            cur_step=int(row['step'])

        if 'epoch' in row and row['epoch']!='':
            cur_epoch=int(row['epoch'])

        if 'lr-Adadelta' in row and row['lr-Adadelta']!='':
            if cur_lr!=float(row['lr-Adadelta']):
                lr_change.append(cur_step)
                cur_lr=float(row['lr-Adadelta'])

        if 'val_loss' in row and row['val_loss']!='':
            cur_vloss=float(row['val_loss'])
            
        if 'cer' in row and row['cer']!='':
            cur_cer=float(row['cer'])

        if 'train_loss' in row and row['train_loss']!='':
            cur_tloss=float(row['train_loss'])

        if 'primus_cer' in row and row['primus_cer']!='':
            primus_cer=float(row['primus_cer'])

        if 'crohme_cer' in row and row['crohme_cer']!='':
            crohme_cer=float(row['crohme_cer'])

        if 'didi_cer' in row and row['didi_cer']!='':
            didi_cer=float(row['didi_cer'])

        if 'didi_no_text_cer' in row and row['didi_no_text_cer']!='':
            didi_no_text_cer=float(row['didi_no_text_cer'])

        if 'table_bank_cer' in row and row['table_bank_cer']!='':
            table_bank_cer=float(row['table_bank_cer'])

        if 'zinc_cer' in row and row['zinc_cer']!='':
            zinc_cer=float(row['zinc_cer'])

        if 'train_loss' in row and row['train_loss']!='':
            needed_data.append([cur_step,cur_lr,cur_epoch,cur_tloss,primus_cer,crohme_cer,didi_cer,didi_no_text_cer,table_bank_cer,zinc_cer])




step_list=[data[0] for data in needed_data]
# train_loss_list=[data[3] for data in needed_data]
primus_cer_list=[data[4] for data in needed_data]
crohme_cer_list=[data[5] for data in needed_data]
didi_cer_list=[data[6] for data in needed_data]
didi_no_text_cer_list=[data[7] for data in needed_data]
table_bank_cer_list=[data[8] for data in needed_data]
zinc_cer_list=[data[9] for data in needed_data]

# 绘制训练损失函数图像
plt.figure(figsize=(10, 6))
# import pdb; pdb.set_trace()
plt.plot(step_list, primus_cer_list, label='primus', marker='o', linestyle='-')
plt.plot(step_list, crohme_cer_list, label='crohme', marker='o', linestyle='-')
plt.plot(step_list, didi_cer_list, label='didi', marker='o', linestyle='-')
plt.plot(step_list, didi_no_text_cer_list, label='didi_no_text', marker='o', linestyle='-')
plt.plot(step_list, table_bank_cer_list, label='table_bank', marker='o', linestyle='-')
plt.plot(step_list, zinc_cer_list, label='zinc', marker='o', linestyle='-')
plt.xlabel('Step')
plt.ylabel('CER')
plt.title('CER Over Steps')
# 在指定的step处绘制竖线
for step in lr_change:
    plt.axvline(x=step, color='r', linestyle='--')

plt.legend()
# plt.grid(True)

# 显示图像
plt.show()



