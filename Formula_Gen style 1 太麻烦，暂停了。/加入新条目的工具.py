from typing import List

#total_list:List[tuple[str, int]] = []
total_list:List[tuple[str, int]] = [("test1", 0),("testtest1", 1),("aa1a", 2),("b1bb", 3),("1ccc", 4),]
len_of_total = len(total_list)

new_list : List[str] = ["test", "testtest", "aaa", "bbb", "ccc", ]

for i in range(len(new_list)):
    entry = new_list[i]
    total_list.append((entry,len_of_total+i))
    pass

#check it a bit.
#duplication:
temp:List[str] = []
for pair in total_list:
    temp.append(pair[0])
    pass
temp.sort()
for i in range(len(temp)-1):
    if temp[i] == temp[i+1]:
        raise Exception("{} is duplicated.".format(temp[i]))
del temp

#index is correct?
last_index = 0
for i in range(1,len(total_list)):
    index = total_list[i][1]
    if last_index+1!=index:
        raise Exception("Index at {} is not correct".format(index))
    last_index = index
    pass

#now, it's correct, let's save the new total list into a file.
ENTRY_PER_LINE = 5
with open("total list output.txt",mode="x") as f:
    f.write("total_list:List[tuple[str, int]] = [\n")
    for i in range(len(total_list)):
        pair = total_list[i]
        f.write("(\"{}\", {}),".format(pair[0], pair[1]))
        if i%ENTRY_PER_LINE == ENTRY_PER_LINE-1:
            f.write("\n")
            pass
        pass
    f.write("]")
    pass

 

