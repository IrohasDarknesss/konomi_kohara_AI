import os
import pyopenjtalk

# convert transcript.txt
in_path = 'filelists/kohara.txt'
out_path = 'filelists/transcript_kohara.txt'
output = []
with open(in_path) as f:
    lines = f.readlines()
    for line in lines:
        strs = line.split(':')
        strs[1] = pyopenjtalk.g2p(strs[1], kana=False)
        strs[1] = strs[1].replace('pau',',')
        strs[1] = strs[1].replace(' ','')
        strs[1] = strs[1] + '.'
        output.append('wav/'+strs[0]+'.wav|'+strs[1]+'\n')

with open(out_path, 'w') as f:
    f.writelines(output)