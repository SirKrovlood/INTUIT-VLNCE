import matplotlib.pyplot as plt
import numpy as np

f = open("da-fineyune-slurm-49602153.out", "r" , encoding="utf8")
rlines = f.readlines()
f.close()

X = 4000* np.arange(1,11)
print(X)

x_r = np.arange(X.sum())

tr_loss = np.empty(X.sum())
ac_loss = np.empty(X.sum())
aux_loss = np.empty(X.sum())

b = 0

for i in range(len(rlines)):
    if "train_loss:" in rlines[i]:
        tr_loss[b] = float(rlines[i].split("train_loss:")[1])
        ac_loss[b] = float(rlines[i+1].split("train_action_loss:")[1])
        aux_loss[b] = float(rlines[i+2].split("train_aux_loss:")[1])
        b += 1
        
print(tr_loss[-1])

print("aux_loss - min:", np.min(aux_loss), ", max:", np.max(aux_loss), ", mean:", np.mean(aux_loss))

#plt.axvline(14700*44, color='purple', label='best checkpoint', lw=3)
plt.scatter(x_r, tr_loss, vmax=1.75, s=0.2, marker='.')
plt.xlim(0, X.sum()+1)
#plt.ylim(0, 1.75)

ticks = np.empty(X.size)
ticks[0] = X[0]

for i in range(1,10):
    ticks[i] = ticks[i-1] + X[i]

best = int(ticks[0] + X[2]//2)

plt.xticks(ticks)
#plt.xticks( np.concatenate((ticks, np.array([best]) )) ) 


for i in range(10):
    plt.axvline(ticks[i], color='black')

plt.axvline(best, color='purple', label='best checkpoint, after '+str(best)+" batches", lw=3)
#for i in range(9):
#    plt.axvline(14700*5*i, color='black', label='5 epochs have passed')

plt.ylabel('loss value')
plt.xlabel('processed batches')

plt.legend()

plt.show()
