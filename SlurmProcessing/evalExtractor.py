f = open("eval1st-slurm-49559215.out", "r")
rlines = f.readlines()
f.close()

with open('1steval-results.txt', 'w') as f:
    
    for line in rlines:
        if "=======current_ckpt:" in line or "Episodes evaluated:" in line or "Average episode" in line:
            f.write(line)
    


f = open("1steval-results.txt", "r")
rlines = f.readlines()
f.close()

i = 0

bigNDTW = 0
bigInd = 0

while i*11 < len(rlines):
    #print(rlines[i*11+4])
    #tmp = float(rlines[i*11+4].split("spl:")[1])
    tmp = float(rlines[i*11+5].split("ndtw:")[1])

    #print(tmp)
    if tmp >= bigNDTW:
        bigInd = i
        bigNDTW = tmp
        print(bigInd, bigNDTW)
    i += 1
        
print("Best performing checkpoint is:", bigInd,", nDTW:", bigNDTW)


f = open("da-eval-7500-slurm-49650736.out", "r")
rlines = f.readlines()
f.close()

with open('2ndeval-results.txt', 'w') as f:
    for line in rlines:
        if "=======current_ckpt:" in line or "Episodes evaluated:" in line or "Average episode" in line:
            f.write(line)

f = open("da-eval-500-slurm-49656877.out", "r")
rlines = f.readlines()
f.close()

with open('2ndeval-results.txt', 'a') as f:
    
    for line in rlines:
        if ("=======current_ckpt:" in line or "Episodes evaluated:" in line or "Average episode" in line) and "/ckpt.13.pth"not in line:
            f.write(line)
            
f = open("2ndeval-results.txt", "r")
rlines = f.readlines()
f.close()

i = 0

bigNDTW = 0
bigInd = 0

while i*11 < len(rlines):
    #print(rlines[i*11+4])
    #tmp = float(rlines[i*11+4].split("spl:")[1])
    tmp = float(rlines[i*11+5].split("ndtw:")[1])
    #print(tmp)
    if tmp >= bigNDTW:
        bigInd = i
        bigNDTW = tmp
        print(bigInd, bigNDTW)
    i += 1
        
print("Best performing DA finetuned checkpoint is:", bigInd,", nDTW:", bigNDTW)