import matplotlib.pyplot as plt
import numpy as np

f = open("train-update-agent-slurm-49528030.out", "r" , encoding="utf8")
rlines = f.readlines()
f.close()

tr_loss = np.empty(14700*45)
ac_loss = np.empty(14700*45)
aux_loss = np.empty(14700*45)

b = 0

for i in range(len(rlines)):
    if "train_loss:" in rlines[i]:
        tr_loss[b] = float(rlines[i].split("train_loss:")[1])
        ac_loss[b] = float(rlines[i+1].split("train_action_loss:")[1])
        aux_loss[b] = float(rlines[i+2].split("train_aux_loss:")[1])
        b += 1
        
print(tr_loss[-1])

print("aux_loss - min:", np.min(aux_loss), ", max:", np.max(aux_loss), ", mean:", np.mean(aux_loss))

plt.axvline(14700*44, color='purple', label='best checkpoint', lw=3)
plt.scatter(np.arange(14700*45), tr_loss, vmax=1.75, s=0.2, marker='.')
plt.xlim(0, 14700*45+1)
plt.ylim(0, 1.75)

print(np.arange(0, 14700*44, 14700*5))
print(np.array([14700*44]))
print(np.concatenate((np.arange(0, 14700*44, 14700*5), np.array([14700*44])) ))

plt.xticks( np.concatenate((np.arange(0, 14700*44, 14700*5), np.array([14700*44]) )) ) 
plt.legend()

for i in range(9):
    plt.axvline(14700*5*i, color='black', label='5 epochs have passed')

plt.ylabel('loss value')
plt.xlabel('processed batches')
plt.show()
