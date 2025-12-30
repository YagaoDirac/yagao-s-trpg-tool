from copy import deepcopy 
import random
import torch


def calc(input:list[int], extra_memory_len:int, total_epoch:int)->list[int]:

    memory:list[int] = deepcopy(input)
    memory.extend([0]*extra_memory_len)

    for ii in range(total_epoch):
        condition_count_with_extra = int(ii*0.05+0.5)
        condition_count_with_input = int(ii*0.05+1)
        boundary_max = int(ii*0.07+1.5)
        modify_by_max = int(ii*0.06+1)#at least 1, this is the up bound
        still_needed = True
        
        #<all the conditions>
        for _extra_test_index in range(condition_count_with_extra):
            test_index = random.randint(input.__len__(), memory.__len__()-1)
            boundary = random.randint(-boundary_max, boundary_max)
            gt_or_lt = random.random()<0.5
            if gt_or_lt and memory[test_index]<=boundary:
                still_needed = False
                break
            if (not gt_or_lt) and memory[test_index]>=boundary:
                still_needed = False
                break
            pass#for 
        if still_needed:
            for _input_test_index in range(condition_count_with_input):
                test_index = random.randint(0, input.__len__()-1)
                boundary = random.randint(-boundary_max, boundary_max)
                gt_or_lt = random.random()<0.5
                if gt_or_lt and memory[test_index]<=boundary:
                    still_needed = False
                    break
                if (not gt_or_lt) and memory[test_index]>=boundary:
                    still_needed = False
                    break
                pass#for 
            pass#still needed
        #</all the conditions>

        if still_needed:
            modify_by = random.randint(1, modify_by_max)
            if random.random()<0.5:
                modify_by = -modify_by
                pass
            memory[random.randint(0, memory.__len__()-1)] += modify_by
            pass#still needed
        
        pass
    return memory
            
if "test" and True:
    test_count = 10000
    _temp_tensor = torch.empty(size=[test_count,5], dtype=torch.float32)
    for ii in range(test_count):
        input = torch.randint(size=[2], low=-2,high=3).tolist()
        output = calc(input=input, extra_memory_len=3, total_epoch=100)
        _temp_tensor[ii] = torch.tensor(output, dtype=torch.float32)
        pass
    print(f"{_temp_tensor.mean(dim=0)}, std:{_temp_tensor.std(dim=0)}")
    pass
    