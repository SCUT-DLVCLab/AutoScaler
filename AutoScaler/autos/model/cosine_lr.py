import math
import numpy as np
import matplotlib.pyplot as plt

def cosine_schedule3(step):
    decay=0.4
    max_lr=1.0
    math_stone=[(0,0.5e5),(0.5e5,6e5),(6e5,6.5e5),(6.5e5,12e5),(12e5,12.5e5),(12.5e5,15e5)]
    phase=0
    mag=1
    if math_stone[0][0]<step<=math_stone[0][1]:
        cur_lr=(max_lr/math_stone[0][1])*step
    elif math_stone[1][0]<step<=math_stone[1][1]:
        phase = (step-math_stone[1][0])/11e5
        mag = 1
        cur_lr=(max_lr*mag*math.cos(2*math.pi * phase) + 1)/2

    elif math_stone[2][0]<step<=math_stone[2][1]:
        mag = 1 * decay
        cur_lr = (max_lr * mag / (math_stone[2][1]-math_stone[2][0])) * (step-math_stone[2][0])
    elif math_stone[3][0]<step<=math_stone[3][1]:
        phase = (step-math_stone[3][0])/11e5
        mag=1 * decay
        cur_lr = (max_lr * mag * math.cos(2 * math.pi * phase) + mag) / 2

    elif math_stone[4][0]<step<=math_stone[4][1]:
        mag = 1 * decay * decay
        cur_lr = (max_lr * mag / (math_stone[4][1]-math_stone[4][0])) * (step-math_stone[4][0])
    elif math_stone[5][0]<step<=math_stone[5][1]:
        phase = (step-math_stone[5][0])/5e5
        mag = 1 * decay * decay
        cur_lr = (max_lr * mag * math.cos(2 * math.pi * phase) + mag) / 2
    else:
        cur_lr = 1e-6
    return cur_lr

if __name__=='__main__':
    steps = np.linspace(0, 20e5, 3000)  # 增加采样点到3000个
    learning_rates = [cosine_schedule3(step) for step in steps]
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates, 'b-', label='Learning Rate')

    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Learning Rate Schedule')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    plt.legend()
    plt.tight_layout()
    plt.show()