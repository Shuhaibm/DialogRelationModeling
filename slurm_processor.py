import sys

filename = sys.argv[1]

file = open(filename, 'r')

empty_lines = ["\n", "                                                     \n",
               "                                                 \n",
               "                                                    \n"]
skip_lines = ["it/s]\n", "[A", "it/s]  \n"]

for line in file.readlines():
    skip = False
    for empty_line in empty_lines:
        if line == empty_line: skip = True
    for skip_line in skip_lines:
        if skip_line in line: skip = True
    
    if skip: continue
    print(line)

