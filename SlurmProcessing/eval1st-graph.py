import matplotlib.pyplot as plt
import numpy as np

f = open("eval1st-slurm-49559215.out", "r" , encoding="utf8")
rlines = f.readlines()
f.close()

ne = np.zeros(45)
sr = np.zeros(45)
spl = np.zeros(45)
ndtw = np.zeros(45)
sdtw = np.zeros(45)
path_length = np.zeros(45)
oracle_success = np.zeros(45)
steps_taken = np.zeros(45)
way_accuracy = np.zeros(45)

b = 0
for i in range(len(rlines)):
    if "Episodes evaluated:" in rlines[i]:
        ne[b] = float(rlines[i+1].split("distance_to_goal:")[1])
        sr[b] = float(rlines[i+2].split("success:")[1])
        spl[b] = float(rlines[i+3].split("spl:")[1])
        
        ndtw[b] = float(rlines[i+4].split("ndtw:")[1])
        sdtw[b] = float(rlines[i+5].split("sdtw:")[1])
        path_length[b] = float(rlines[i+6].split("path_length:")[1])
        
        oracle_success[b] = float(rlines[i+7].split("oracle_success:")[1])
        steps_taken[b] = float(rlines[i+8].split("steps_taken:")[1])
        way_accuracy[b] = float(rlines[i+9].split("way_accuracy:")[1])
        
        b += 1
        

f = open("valseen-1st-slurm-49657711.out", "r" , encoding="utf8")
rlines = f.readlines()
f.close()

ne_seen = np.zeros(45)
sr_seen = np.zeros(45)
spl_seen = np.zeros(45)
ndtw_seen = np.zeros(45)
sdtw_seen = np.zeros(45)
path_length_seen = np.zeros(45)
oracle_success_seen = np.zeros(45)
steps_taken_seen = np.zeros(45)
way_accuracy_seen = np.zeros(45)

b = 0
for i in range(len(rlines)):
    if "Episodes evaluated:" in rlines[i]:
        ne_seen[b] = float(rlines[i+1].split("distance_to_goal:")[1])
        sr_seen[b] = float(rlines[i+2].split("success:")[1])
        spl_seen[b] = float(rlines[i+3].split("spl:")[1])
        
        ndtw_seen[b] = float(rlines[i+4].split("ndtw:")[1])
        sdtw_seen[b] = float(rlines[i+5].split("sdtw:")[1])
        path_length_seen[b] = float(rlines[i+6].split("path_length:")[1])
        
        oracle_success_seen[b] = float(rlines[i+7].split("oracle_success:")[1])
        steps_taken_seen[b] = float(rlines[i+8].split("steps_taken:")[1])
        way_accuracy_seen[b] = float(rlines[i+9].split("way_accuracy:")[1])
        
        b += 1

############
plt.axvline(43, color='purple', label='best checkpoint', lw=3)
plt.scatter(np.arange(45), sr, marker='o', label='val_unseen SR')
plt.plot( sr)

plt.scatter(np.arange(45), oracle_success, marker='^', label='val_unseen OS')
plt.plot( oracle_success)

plt.scatter(np.arange(45), sr_seen, marker='p', label='val_seen SR')
plt.plot(sr_seen)

plt.scatter(np.arange(45), oracle_success_seen, marker='v', label='val_seen OS')
plt.plot( oracle_success_seen)

plt.xlim(-1, 46)

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.ylabel('rate value')
plt.xlabel('checkpoint number')

plt.legend()
plt.show()
###################################

plt.axvline(43, color='purple', label='best checkpoint', lw=3)

plt.scatter(np.arange(45), ndtw, marker='o', label='val_unseen nDTW')
plt.plot( ndtw)
plt.scatter(np.arange(45), sdtw, marker='^', label='val_unseen sDTW')
plt.plot( sdtw)

plt.scatter(np.arange(45), ndtw_seen, marker='p', label='val_seen nDTW')
plt.plot( ndtw_seen)
plt.scatter(np.arange(45), sdtw_seen, marker='v', label='val_seen sDTW')
plt.plot( sdtw_seen)

plt.xlim(-1, 46)

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.ylabel('rate value')
plt.xlabel('checkpoint number')

plt.legend()
plt.show()

########################################
plt.axvline(43, color='purple', label='best checkpoint', lw=3)

plt.scatter(np.arange(45), spl, marker='^', label='val_unseen SPL')
plt.plot( spl)

plt.scatter(np.arange(45), spl_seen, marker='v', label='val_seen SPL')
plt.plot( spl_seen)

plt.xlim(-1, 46)

plt.ylabel('rate value')
plt.xlabel('checkpoint number')

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.legend()
plt.show()

########################################
plt.axvline(43, color='purple', label='best checkpoint', lw=3)

plt.scatter(np.arange(45), ne, marker='^', label='val_unseen NE')
plt.plot( ne)

plt.scatter(np.arange(45), ne_seen, marker='v', label='val_seen NE')
plt.plot( ne_seen)

plt.xlim(-1, 46)

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.ylabel('meters')
plt.xlabel('checkpoint number')

plt.legend()
plt.show()

########################################
plt.axvline(43, color='purple', label='best checkpoint', lw=3)

plt.scatter(np.arange(45), path_length , marker='^', label='val_unseen TL')
plt.plot( path_length)

plt.scatter(np.arange(45), path_length_seen, marker='v', label='val_seen TL')
plt.plot( path_length_seen)

plt.xlim(-1, 46)

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.ylabel('meters')
plt.xlabel('checkpoint number')

plt.legend()
plt.show()

########################################
plt.axvline(43, color='purple', label='best checkpoint', lw=3)

plt.scatter(np.arange(45), steps_taken , marker='^', label='val_unseen Steps Taken')
plt.plot( steps_taken)

plt.scatter(np.arange(45), steps_taken_seen, marker='v', label='val_seen Steps Taken')
plt.plot( steps_taken_seen)

plt.xlim(-1, 46)

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.ylabel('number of steps')
plt.xlabel('checkpoint number')

plt.legend()
plt.show()

########################################
plt.axvline(43, color='purple', label='best checkpoint', lw=3)

plt.scatter(np.arange(45), way_accuracy , marker='^', label='val_unseen WA')
plt.plot( way_accuracy)

plt.scatter(np.arange(45), way_accuracy_seen, marker='v', label='val_seen WA')
plt.plot( way_accuracy_seen)

plt.xlim(-1, 46)

plt.xticks(list(plt.xticks()[0][1:-1]) + [43])

plt.ylabel('fraction')
plt.xlabel('checkpoint number')

plt.legend()
plt.show()
