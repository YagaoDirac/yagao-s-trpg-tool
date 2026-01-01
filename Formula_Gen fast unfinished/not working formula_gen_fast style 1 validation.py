from copy import deepcopy 
import random
import torch



简单说一下现在的问题。
就是我很在意的，稍微的改变输入，应该得到相似的输出，这一条用现在这个写法是无法实现的。
想了一下可能还是神经网络比较现实。
那目测要等我的新版的softmax写好。




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
        condition_count_with_input__max = int(ii/randcfg.total_epoch*\
                    randcfg.condition_count_with_input__max)+1
        #condition_count_with_extra = int(ii*randcfg.condition_count_with_extra__k+randcfg.condition_count_with_extra__b)
        condition_count_with_extra__max = int(ii/randcfg.total_epoch*\
                    (randcfg.condition_count_with_extra__max+1))
        boundary_max = int(ii*randcfg.boundary_max__k+randcfg.boundary_max__b)
        modify_by_max__float = ii*randcfg.modify_by_max__k+randcfg.modify_by_max__b
        
        #fg.boundary_max__k+randcfg.boundary_max__b
        
        pass_count = 0
        fail_count = 0
        
        if _debug__return_log:
            _log_for_epoch = ""
            pass
        
        #<all the conditions>
        condition_count_with_extra = random.randint(0,condition_count_with_extra__max)
        for _extra_test_index in range(condition_count_with_extra):
            test_index = random.randint(input.__len__(), memory.__len__()-1)
            boundary = random.randint(-boundary_max, boundary_max)
            gt_or_lt = random.random()<0.5
            _pass = False
            if gt_or_lt and memory[test_index]>boundary:
                _pass = True
                pass
            if (not gt_or_lt) and memory[test_index]<boundary:
                _pass = True
                pass
            if _pass:
                pass_count+=1
                pass
            else:
                fail_count+=1
                pass
            
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
        assert pass_count+fail_count == condition_count_with_extra
        
        
        condition_count_with_input = random.randint(1,condition_count_with_input__max)
        for _input_test_index in range(condition_count_with_input):
            test_index = random.randint(0, input.__len__()-1)
            boundary = random.randint(-boundary_max, boundary_max)
            gt_or_lt = random.random()<0.5
            
            _pass = False
            if gt_or_lt and memory[test_index]>boundary:
                _pass = True
                pass
            if (not gt_or_lt) and memory[test_index]<boundary:
                _pass = True
                pass
            if _pass:
                pass_count+=1
                pass
            else:
                fail_count+=1
                pass
                
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
        #</all the conditions>


        assert pass_count+fail_count == condition_count_with_input+condition_count_with_extra


        
        _at_least_pass_count = (condition_count_with_input+condition_count_with_extra)//2
        _extra_pass_count = pass_count - _at_least_pass_count
        if _extra_pass_count>0:
            modify_by = random.randint(1, int(modify_by_max__float*_extra_pass_count))
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




if "log" and False:
    randcfg = Random_cfg(300,  3,5,  3.,16,   3.,15.,)
    bound = 10
    input_len = 3
    extra_memory_len = 5
    input_list = torch.randint(size=[input_len], low=-bound,high=bound+1).tolist()
    _temp_output_tuple_ll = calc(input=input_list, extra_memory_len=extra_memory_len, randcfg=randcfg,\
                            _debug__return_log=True)
    _log = _temp_output_tuple_ll[1]
    pass

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

if "test" and False:
    randcfg = Random_cfg(500,  6,9,  3.,16,   3.,15.,)#std:18.6, 17.4.  dim 15 35, bound 10
    # randcfg = Random_cfg(500,  2,2,  4.,20.,   2.,12.,)
    bound = 10
    input_len = 15
    extra_memory_len = 35
    test_count = 1000
    _temp_tensor = torch.empty(size=[test_count,input_len+extra_memory_len], dtype=torch.float32)
    for test_index in range(test_count):
        input_list = torch.randint(size=[input_len], low=-bound,high=bound+1).tolist()
        _temp_output_tuple_ll = calc(input=input_list, extra_memory_len=extra_memory_len, randcfg=randcfg)
        output = _temp_output_tuple_ll[0]
        _temp_tensor[test_index] = torch.tensor(output, dtype=torch.float32)
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




if "test" and True:
    #randcfg = Random_cfg(300,  3,5,  3.,16,   3.,15.,)
    randcfg = Random_cfg(1000,  3,5,  1.,8,   3.,15.,)
    bound = 10
    input_len = 15
    extra_memory_len = 35
    
    
    input_list = torch.randint(size=[input_len], low=-bound,high=bound+1).tolist()
    _temp_output_tuple_ll = calc(input=input_list, extra_memory_len=extra_memory_len, randcfg=randcfg, _debug__return_log = True)
    _log = _temp_output_tuple_ll[1]
    
    
    test_count = 100
    _temp_tensor = torch.empty(size=[test_count,input_len+extra_memory_len], dtype=torch.float32)
    for test_index in range(test_count):
        input_list = torch.randint(size=[input_len], low=-bound,high=bound+1).tolist()
        _temp_output_tuple_ll = calc(input=input_list, extra_memory_len=extra_memory_len, randcfg=randcfg)
        output = _temp_output_tuple_ll[0]
        _temp_tensor[test_index] = torch.tensor(output, dtype=torch.float32)
        pass
    assert _temp_tensor[:,:input_len].shape == torch.Size([test_count, input_len])
    input_mean = _temp_tensor[:,:input_len].mean()
    assert input_mean.abs().item()<0.5
    extra_mean = _temp_tensor[:,input_len:].mean()
    assert extra_mean.abs().item()<0.5
    input_std = _temp_tensor[:,:input_len].std().item()
    extra_std = _temp_tensor[:,input_len:].std().item()
    print(f"input_std={input_std:.3f}, extra_std={extra_std:.3f}")
    
    
    diff_strength__max = 15
    test_time_per_each_case = 20#1000
    _temp_tensor = torch.empty(size=[diff_strength__max, test_time_per_each_case, input_len+extra_memory_len], dtype=torch.float32)
    list_of_test_index = [0]*diff_strength__max
    while True:
        _tensor_of_test_index = torch.tensor(list_of_test_index)
        if _tensor_of_test_index.ge(test_time_per_each_case).all():
            break
        _diff_strength = -1
        ori_input = torch.randint(size=[input_len], low=-bound,high=bound+1)
        comp_input = ori_input.detach().clone()
        #<find the comp_input
        while True:
            _diff_strength = int((ori_input-comp_input).abs().sum().item())
            if _diff_strength>diff_strength__max:#too far, reset
                comp_input = ori_input.detach().clone()
                _diff_strength = 0
                pass
            if comp_input.gt(bound).any() or comp_input.lt(-bound).any():
                comp_input = ori_input.detach().clone()
                _diff_strength = 0
                pass
            if (_diff_strength == 0) or \
                    (list_of_test_index[_diff_strength-1]>=test_time_per_each_case):
                #the case is not needed. Tweak it a lil bit.
                if random.random()<0.5:
                    comp_input[random.randint(0, input_len-1)] += 1
                    pass
                else:
                    comp_input[random.randint(0, input_len-1)] -= 1
                    pass
                continue
            else:#the case is needed
                break
            #notail
            pass#while(inner)
        #</find the comp_input
        assert _diff_strength >=1 and _diff_strength <=diff_strength__max
        
        _temp_output_tuple_ll = calc(input= ori_input.tolist(), extra_memory_len=extra_memory_len, randcfg=randcfg)
        ori_output = _temp_output_tuple_ll[0]
        _temp_output_tuple_ll = calc(input=comp_input.tolist(), extra_memory_len=extra_memory_len, randcfg=randcfg)
        comp_output = _temp_output_tuple_ll[0]
        
        test_index = list_of_test_index[_diff_strength-1]
        _temp_tensor[_diff_strength-1, test_index]  = torch.tensor( ori_output, dtype=torch.float32)
        _temp_tensor[_diff_strength-1, test_index] -= torch.tensor(comp_output, dtype=torch.float32)
        fdsfds = _temp_tensor[_diff_strength-1, test_index]
        #tail
        list_of_test_index[_diff_strength-1] +=1
        pass#while (outter)
        
    
    assert _temp_tensor[:,:,:input_len].shape == torch.Size([diff_strength__max, test_time_per_each_case, input_len])
    input_diff_mean = _temp_tensor[:,:,:input_len].mean().item()
    assert input_diff_mean<0.5
    extra_diff_mean = _temp_tensor[:,:,input_len:].mean().item()
    assert extra_diff_mean<0.5
    
    for _diff_strength in range(1,diff_strength__max+1):
        #input_std = _temp_tensor[_diff_strength-1,:,:input_len].std().item()
        #extra_std = _temp_tensor[_diff_strength-1,:,input_len:].std().item()
        input_std = _temp_tensor[_diff_strength-1,:,:input_len].abs().mean().item()
        extra_std = _temp_tensor[_diff_strength-1,:,input_len:].abs().mean().item()
        print( f"_diff_strength={_diff_strength}, input_diff_abs_mean={input_std:.3f}, extra_diff_abs_mean={extra_std:.3f}")
        pass
    pass