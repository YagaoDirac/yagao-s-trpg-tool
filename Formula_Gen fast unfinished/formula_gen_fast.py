from copy import deepcopy 
import random
import torch



# for ii in range(10):
#     condition_count_with_input = int(ii/10*3)+1
#     print(condition_count_with_input)
#     pass





class Random_cfg():
    total_epoch:int
    # condition_count_with_input__k:float
    # condition_count_with_input__b:float
    condition_count_with_input__max:int
    
    # condition_count_with_extra__k:float
    # condition_count_with_extra__b:float
    condition_count_with_extra__max:int
    
    boundary_max__k:float
    boundary_max__b:float
    modify_by_max__k:float 
    modify_by_max__b:float 
    # modify_bonus__k:float
    # modify_bonus__b:float
        
    def __init__(self, total_epoch:int, \
                #condition_count_with_input__k:float, condition_count_with_input__b:float, \
                condition_count_with_input__max:int, \
                #condition_count_with_extra__k:float, condition_count_with_extra__b:float, \
                condition_count_with_extra__max:int, \
                boundary_max__from:float, boundary_max__to:float, \
                modify_by_max__from:float,  modify_by_max__to:float, \
                #modify_bonus__from:float,  modify_bonus__to:float, \
                    ):
        assert total_epoch>=5
        # assert condition_count_with_input__k>0.
        # assert condition_count_with_input__b>=1.
        assert condition_count_with_input__max>1
        # assert condition_count_with_extra__k>0.
        # assert condition_count_with_extra__b>0.
        assert condition_count_with_extra__max>=1
        assert boundary_max__from>=1.
        assert boundary_max__to>boundary_max__from
        assert modify_by_max__from>=1.
        assert modify_by_max__to>modify_by_max__from
        # assert modify_bonus__from>0.
        # assert modify_bonus__to>modify_bonus__from
        
        self.total_epoch = total_epoch
        # self.condition_count_with_input__k = condition_count_with_input__k
        # self.condition_count_with_input__b = condition_count_with_input__b
        self.condition_count_with_input__max = condition_count_with_input__max
        # self.condition_count_with_extra__k = condition_count_with_extra__k
        # self.condition_count_with_extra__b = condition_count_with_extra__b
        self.condition_count_with_extra__max = condition_count_with_extra__max
        self.boundary_max__k = (boundary_max__to-boundary_max__from)/total_epoch
        self.boundary_max__b = boundary_max__from
        self.modify_by_max__k = (modify_by_max__to-modify_by_max__from)/total_epoch
        self.modify_by_max__b = modify_by_max__from
        # self.modify_bonus__k = (modify_bonus__to-modify_bonus__from)/total_epoch
        # self.modify_bonus__b = modify_bonus__from
        pass
    
    #@staticmethod
    
    


def calc(input:list[int], extra_memory_len:int, randcfg:Random_cfg, \
                    _debug__return_log=False)->tuple[list[int], list[str]|None]:
    if _debug__return_log:
        _log:list[str]|None = []
        pass
    else:
        _log = None
        pass
        
    memory:list[int] = deepcopy(input)
    memory.extend([0]*extra_memory_len)

    for ii in range(randcfg.total_epoch):
        #condition_count_with_input = int(ii*randcfg.condition_count_with_input__k+randcfg.condition_count_with_input__b)
        condition_count_with_input = int(ii/randcfg.total_epoch*\
                    randcfg.condition_count_with_input__max)+1
        #condition_count_with_extra = int(ii*randcfg.condition_count_with_extra__k+randcfg.condition_count_with_extra__b)
        condition_count_with_extra = int(ii/randcfg.total_epoch*\
                    (randcfg.condition_count_with_extra__max+1))
        boundary_max = int(ii*randcfg.boundary_max__k+randcfg.boundary_max__b)
        modify_by_max = int(ii*randcfg.modify_by_max__k+randcfg.modify_by_max__b)
        
        #fg.boundary_max__k+randcfg.boundary_max__b
        
        still_needed = True
        
        if _debug__return_log:
            _log_for_epoch = ""
            pass
        
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
            
            if _debug__return_log:
                _log_for_epoch += f"e"
                if gt_or_lt:
                    _log_for_epoch += ">"
                    pass
                else:
                    _log_for_epoch += "<"
                    pass
                _log_for_epoch += f"{boundary},"
                pass
            
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
                    
                if _debug__return_log:
                    _log_for_epoch += f"i"
                    if gt_or_lt:
                        _log_for_epoch += ">"
                        pass
                    else:
                        _log_for_epoch += "<"
                        pass
                    _log_for_epoch += f"{boundary},"
                    pass
                    
                pass#for 
            pass#still needed
        #</all the conditions>

        if still_needed:
            total_cond_count = condition_count_with_input+condition_count_with_extra
            modify_by = random.randint(1, modify_by_max*total_cond_count)
            if random.random()<0.5:
                modify_by = -modify_by
                pass
            memory[random.randint(0, memory.__len__()-1)] += modify_by
            
            if _log is not None:
                _log_for_epoch = f"ep{ii}  {_log_for_epoch}  modified:{modify_by}"
                _log.append(_log_for_epoch)
                pass
            
            pass#still needed
        
        pass
    return memory, _log
            
            
            
            
if "bound to init" and False:
    # result is basically std_input is 0.6*bound.
    #bound = 5
    for bound in range(2,10):
        input = torch.randint(size=[15000], low=-bound,high=bound+1).to(torch.float32)
        input_mean = input.mean()
        assert input_mean.abs().item()<0.1
        input_std = input.to(torch.float32).std().item()
        print(f"bound={bound:2}  input_std={input_std:.3f}")
        pass
    pass

if "log" and False:
    randcfg = Random_cfg(30,  2,1,  3.,15,   3.,7.,)
    bound = 5
    input_len = 3
    extra_memory_len = 5
    input = torch.randint(size=[input_len], low=-bound,high=bound+1).tolist()
    output = calc(input=input, extra_memory_len=extra_memory_len, randcfg=randcfg,\
                            _debug__return_log=True)
    _log = output[1]
    pass



if "test" and True:
    #randcfg = Random_cfg(1000,  0.05,1.,  0.05,0.5,  0.07,1.5,  0.06,1.)
    #randcfg = Random_cfg(1000,  0.05,1.,  0.05,0.5,  0.07,1.5,  0.6,1.)
    randcfg = Random_cfg(500,  2,2,  4.,20.,   2.,12.,)#   1.5,10.)
    bound = 10
    test_count = 1000
    input_len = 15
    extra_memory_len = 35
    _temp_tensor = torch.empty(size=[test_count,input_len+extra_memory_len], dtype=torch.float32)
    for ii in range(test_count):
        input = torch.randint(size=[input_len], low=-bound,high=bound+1).tolist()
        output = calc(input=input, extra_memory_len=extra_memory_len, randcfg=randcfg)
        _temp_tensor[ii] = torch.tensor(output[0], dtype=torch.float32)
        pass
    assert _temp_tensor[:,:input_len].shape == torch.Size([test_count, input_len])
    input_mean = _temp_tensor[:,:input_len].mean()
    assert input_mean.abs().item()<0.3
    extra_mean = _temp_tensor[:,input_len:].mean()
    assert extra_mean.abs().item()<0.3
    input_std = _temp_tensor[:,:input_len].std().item()
    extra_std = _temp_tensor[:,input_len:].std().item()
    print(f"input_std={input_std:.3f}, extra_std={extra_std:.3f}")
    pass
    