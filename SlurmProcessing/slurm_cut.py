f = open("train-update-agent-slurm-49528030.out", "r" , encoding="utf8")
rlines = f.readlines()
f.close()

l = len(rlines)

f = open("train-update-agent-slurm-49528030pt1.out", "w" , encoding="utf8")
f.writelines(rlines[:l//3])
f.close()

f = open("train-update-agent-slurm-49528030pt2.out", "w" , encoding="utf8")
f.writelines(rlines[l//3:2*l//3])
f.close()

f = open("train-update-agent-slurm-49528030pt3.out", "w" , encoding="utf8")
f.writelines(rlines[2*l//3:])
f.close()

