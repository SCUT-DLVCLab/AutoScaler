import csv
import os
import matplotlib.pyplot as plt

if os.path.exists('metrics.csv'):
    path='metrics.csv'
else:
    v=input('input version:\n')
    path=f'./lightning_logs/version_{v}/metrics.csv'
    
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    cur_lr=float('Inf')
    cur_step=0
    cur_loss=float('Inf')
    cur_cer=float('Inf')
    cur_epoch=0
    cur_vloss=float('Inf')
    cur_tloss=float('Inf')
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

        if 'train_loss' in row and row['train_loss']!='':
            needed_data.append([cur_step,cur_lr,cur_epoch,cur_tloss])


step_list=[data[0] for data in needed_data]
train_loss_list=[data[3] for data in needed_data]
# 绘制训练损失函数图像
plt.figure(figsize=(10, 6))
plt.plot(step_list, train_loss_list, label='Train Loss', marker='o', linestyle='-')
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.title('Training Loss Over Steps')
# 在指定的step处绘制竖线
for step in lr_change:
    plt.axvline(x=step, color='r', linestyle='--')

plt.legend()
# plt.grid(True)

# 显示图像
plt.show()



