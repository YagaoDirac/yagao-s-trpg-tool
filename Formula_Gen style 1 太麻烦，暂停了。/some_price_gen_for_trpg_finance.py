from typing import List, Any
import math
import json
import torch
from v2.v2_pure_code import MirrorWithGramo


class the_model(torch.nn.Module):
    def __init__(self, in_out_features: int, ref_feature: int, bias: bool = True, \
                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mid_width = 32
        self.in_out_features = in_out_features
        self.ref_feature = ref_feature
        self.mig1 = MirrorWithGramo(ref_feature+in_out_features,mid_width)
        self.mig2 = MirrorWithGramo(mid_width,  mid_width)
        self.mig3 = MirrorWithGramo(mid_width,  mid_width)
        self.mig4 = MirrorWithGramo(mid_width,  mid_width)
        self.mig5 = MirrorWithGramo(mid_width,  in_out_features)
        pass

    def forward(self, x:torch.Tensor)->torch.Tensor:
        input = x
        x = self.mig1(x)
        x = self.mig2(x)
        x = self.mig3(x)
        x = self.mig4(x)
        x = self.mig5(x)
        return input[:self.in_out_features]+x
    
    # def before_optim_step(self)->None:
        # with torch.interference_mode():
        # self.mig1.bias.grad*=1.
        # self.mig2.bias.grad*=1.
        # self.mig3.bias.grad*=1.
        # self.mig4.bias.grad*=1.
        # self.mig5.bias.grad*=1.
        #pass
    pass



class ref_factor_entry_info:
    def __init__(self, name:str, index:int) -> None:
        self.name = name
        self.index = index
        pass
    # def order_too_big(self, input:float)->bool:
    #     return input>self.up_order
    # def order_too_small(self, input:float)->bool:
    #     return input<self.low_order
    # def order_of_mag_to_real(self, input:float)->float:
    #     return math.pow(self.base, input)
class ref_factor_entry_container:
    as_torch_tensor:torch.Tensor
    def __init__(self, data:List[ref_factor_entry_info] = []) -> None:
        self.raw_entry_data = data
        self.dirty = True
        self.info_data:List[tuple[str, float]] = []
        pass
    def append_entry(self, entry:ref_factor_entry_info)->None:
        self.dirty = True
        self.raw_entry_data.append(entry)
        pass
    def update(self)->None:
        if len(self.raw_entry_data) == 0:
            self.dirty = False
            return
        #check it a bit. 
        #No duplication
        for i in range(len(self.raw_entry_data)-1):
            name1 = self.raw_entry_data[i].name
            index1 = self.raw_entry_data[i].index
            for ii in range(i+1, len(self.raw_entry_data)):
                if name1 == self.raw_entry_data[i+1].name:
                    raise Exception("The name: {} is duplicated.".format(name1))
                if index1 == self.raw_entry_data[i+1].index:
                    raise Exception("The index: {} is duplicated.".format(index1))
        ### 0 indent (back to function)
        self.dirty = False
        pass

    def add_info(self, name:str, strength:float, overwrite:bool = True):
        # name MUST be in some entry
        for entry in self.raw_entry_data:
            if name == entry.name:
                break
        ### 0 indent (back to function)
        else:
            raise Exception("{} is not found in entry info. Add the entry first.".format(name))
        
        for i, info_entry in enumerate(self.info_data):
            if info_entry[0] == name:
                if overwrite:
                    self.info_data[i] = (name, strength)
                else:
                    self.info_data[i] = (name, self.info_data[i][1]+strength)
                    pass
                break
            pass
        else:
            self.info_data.append((name, strength))
        pass    
    def add_info_array(self, data:List[tuple[str, float]])->None:
        for item in data:
            self.add_info(item[0], item[1])
            pass
        pass

    def get_torch_tensor(self)->torch.Tensor:
        if self.dirty:
            self.update()

        max_index = 0
        for entry in self.raw_entry_data:
            if max_index<entry.index:
                max_index = entry.index
        ### 0 indent (back to function)

        result:torch.Tensor = torch.zeros([max_index+1])
        for info in self.info_data:
            for entry in self.raw_entry_data:
                if info[0] == entry.name:
                    result[entry.index] = info[1]
        ### 0 indent (back to function)
        return result







# class item_price_info_entry:
#     def __init__(self, name:str, index:int, low_order_of_mag:float, up_order_of_mag:float, base:float = 10) -> None:
#         self.name = name
#         self.index = index
#         self.low_order = low_order_of_mag
#         self.up_order = up_order_of_mag
#         self.base = base
#         pass
#     def order_too_big(self, input:float)->bool:
#         return input>self.up_order
#     def order_too_small(self, input:float)->bool:
#         return input<self.low_order
#     def order_of_mag_to_real(self, input:float)->float:
#         return math.pow(self.base, input)
class item_price_info_container:
    def __init__(self, dim:int, track_steps:int) -> None:
        self.track_steps = track_steps
        self.price_log = torch.ones((track_steps+1,dim))*0.5
        self.price_log_train_with_this = torch.ones((track_steps,dim))*0.5
        self.max_log = torch.ones((1,dim))
        self.min_log = torch.zeros((1,dim))
        self.names = ["unset name" for i in range(dim)]
        pass
    def set_name(self, index:int, name:str)->None:
        self.names[index] = name
        pass
    def set_data_from_model(self, tensor_from_model:torch.Tensor)->None:
        if tensor_from_model.shape != self.price_log.shape:
            raise Exception("The shapes are different. Something must be wrong.")
        
        self.price_log = tensor_from_model
    def needs_training(self)->tuple[bool, str]:
        needs = False
        for i in range(self.price_log.shape[0]):
            if self.price_log[0][i]>self.max_log[0][i]:
                self.price_log_train_with_this[0][i] = self.min_log[0][i]
                needs = True
                pass
            if self.price_log[0][i]<self.min_log[0][i]:
                self.price_log_train_with_this[0][i] = self.max_log[0][i]
                needs = True
                pass
        #reset indent
        if(needs):
            return needs, "train the model a bit with X = self.price_log, Y = self.price_log_train_with_this"
        else:
            return needs, ""





    def get_readable(self, tensor_from_model:torch.Tensor)->List[tuple[str, float]]:
        






data:ref_factor_entry_container = ref_factor_entry_container()
data.append_entry(ref_factor_entry_info("影响因素1",0))
data.append_entry(ref_factor_entry_info("影响因素2",1))
data.append_entry(ref_factor_entry_info("影响因素3",2))
data.append_entry(ref_factor_entry_info("影响因素4",3))
data.append_entry(ref_factor_entry_info("影响因素5",4))
data.add_info("影响因素2",1.)
data.add_info("影响因素2",1.1)
data.add_info("影响因素2",1.11,False)
data.add_info("影响因素1",1.)
data.add_info("影响因素3",3.)

#data.add_info("影响因素2333",1.11,False)
ref_factor = data.get_torch_tensor()

b = json.dumps(data.info_data)


c = json.loads(b)
data.info_data = []
data.add_info_array(c)

# d = entry_container()
# d.append_entry(entry_info("影响因素1",0,0.,2.))
# d.append_entry(entry_info("影响因素2",1,0.,2.))
# d.append_entry(entry_info("影响因素3",2,0.,2.))
# d.append_entry(entry_info("影响因素4",3,0.,2.))
# d.append_entry(entry_info("影响因素5",4,0.,2.))



ref_factor = ref_factor.unsqueeze(1)

model = the_model(20,ref_factor.shape[0])
init_data = torch.rand((1, 20,))


jflkds = 456







#lr = 0.00001