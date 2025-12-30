
'''
Known issue.
When using customized backward function, 
2 ways are accepted, the 1st is :
loss = loss_func(some param, torch.tensor(some data, requires_grad = True))
The 2dn way is :
target = torch.tensor(some data)
loss = loss_func(some param, target)
By default, torch.tensor set the requires_grad to False. 
So, some black magic happens behind the scene.

To access the learning rate, the code looks like:
optimizer.param_groups[0]["lr"]
You can also set to this member to change it directly. I believe it's safe.
Most of the basic test are done with SGD for simplicity.

Search for convert_to_plain_Linear function.
This function provides a equivalence plain Linear layer( or the Dense layer in Tensorflow).
You basically want to convert the mirror (both w/ and w/o the gramo enhancement) to a plain Linear one before you do the ONNX part.
The reason is, a mirror is basically 2 Linear, but does the same thing.
Mirror is only good for training, but 1x slower in predicting while needs 1x more memory.
'''


from typing import Any, List
import torch
import math

__all__ = [
    'GradientModification',
    'MirrorLayer',
    'MirrorWithGramo',
    'GradientModificationFunction', #Should I expose this?
    'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
    ]

class GradientModificationFunction(torch.autograd.Function):
    r'''input param list:
    x:torch.Tensor,(must be set as require_grad = True)
    learning_rate:float, 
    epi=torch.tensor([1e-12]),
    div_me_when_g_too_small = torch.tensor([1e3])
    
    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_ratio:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        scaling_ratio = args[1]
        epi = args[2]
        div_me_when_g_too_small = args[3]
        # the default values:
        # scaling_ratio = torch.tensor([1.])
        # epi = torch.tensor([0.00001]) 
        # div_me_when_g_too_small = torch.tensor([0.001]) 
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

        ctx.save_for_backward(scaling_ratio, epi, div_me_when_g_too_small)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        scaling_ratio:torch.Tensor
        scaling_ratio, epi, div_me_when_g_too_small = ctx.saved_tensors

        #the shape should only be rank2 with[batch, something]
        # original_shape = g.shape
        # if len(g.shape) == 1:
        #     g = g.unsqueeze(1)
        # protection against div 0    
        length = g.mul(g).sum(dim=1,).sqrt()
        too_small = length.le(epi).to(torch.float32)
        div_me = length*(too_small*-1.+1)+div_me_when_g_too_small*too_small
        div_me = div_me.unsqueeze(dim=1)
        g_out:torch.Tensor = g/div_me
        
        if 1.!=scaling_ratio.item():
            g_out *= scaling_ratio
            pass

        return g_out, None, None, None

    pass  # class




class GradientModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    scaling_ratio:torch.Tensor
    def __init__(self, scaling_ratio:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio]), requires_grad=False)
        self.epi=torch.nn.Parameter(torch.tensor([epi], requires_grad=False))
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small], requires_grad=False))
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if not x.requires_grad:
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction.apply(x, self.scaling_ratio, self.epi, \
                                                   self.div_me_when_g_too_small)
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio], requires_grad=False))
        pass




# This is actually test code. But it looks very useful... 
# I should say, don't use this one. A better one is at the end. You'll love it.
class Linear_gramo(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,\
                            device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features, out_features, bias, device, dtype)
        self.gramo = GradientModification()

    def forward(self, x:torch.tensor)->torch.Tensor:
        #maybe I should handle the shape here.
        x = self.linear(x)
        x = self.gramo(x)
        return x
    



# '''I copied the torch.nn.Linear code and modified it.
# '''

class MirrorLayer(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]

    check torch.nn.Linear for other help
    """
    __constants__ = ['in_features', 'out_features', 'auto_merge_duration']
    in_features: int
    out_features: int
    half_weight: torch.Tensor
    half_weight_mirrored: torch.Tensor

    auto_merge_duration:int
    update_count:int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, auto_merge_duration:int = 20) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.half_weight_mirrored = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        #to keep track of the training.
        self.auto_merge_duration:int = auto_merge_duration
        self.update_count:int = 0
        pass

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.half_weight_mirrored.copy_(self.half_weight)
            pass

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            pass
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.update_count>=self.auto_merge_duration:
            self.update_count = 0
            with torch.no_grad():
                self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
                self.half_weight_mirrored.copy_(self.half_weight)
                pass
            pass

        head1:torch.Tensor = torch.nn.functional.linear(input + 0.5, self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input - 0.5, self.half_weight_mirrored, self.bias)
        return head1+head2

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def convert_to_plain_Linear(self)->torch.nn.Linear:
        has_bias = bool(self.bias)
        result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
        with torch.no_grad():
            result.weight = self.half_weight+self.half_weight_mirrored
            result.bias.copy_(self.bias)
            pass
        return result
    
    def copy_from_plain_Linear(self, intput_linear_layer:torch.nn.Linear)->None:
        with torch.no_grad():
            self.half_weight = intput_linear_layer.weight/2.
            self.half_weight_mirrored.copy_(self.half_weight)
            if intput_linear_layer.bias:
                self.bias.copy_(intput_linear_layer.bias)
                pass
            pass

    pass




# The Mirror part ends.
# Now please welcome, Mirror with Gramo.
# Emmm, somebody please rename this.
# Maybe, Mig?

'''I recommend you use this layer only. Forget about the 2 above. 
This layer is very similar to the previous one.
Basically, if you know how to combine those 2, you can do it yourself.
But mathmatically, you should put the gramo at the very end of mirror. 
In other words, you need 1 gramo for every mirror, not 2.
'''

class MirrorWithGramo(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]

    check torch.nn.Linear for other help
    """
    __constants__ = ['in_features', 'out_features', 'auto_merge_duration']
    in_features: int
    out_features: int
    half_weight: torch.Tensor
    half_weight_mirrored: torch.Tensor

    auto_merge_duration:int
    update_count:int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, auto_merge_duration:int = 20,\
                 scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.half_weight_mirrored = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        #to keep track of the training.
        self.auto_merge_duration:int = auto_merge_duration
        self.update_count:int = 0

        #this layer also needs the info for gramo.
        self.gramo = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
        pass

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.half_weight_mirrored.copy_(self.half_weight)
            pass

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            pass
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.update_count>=self.auto_merge_duration:
            self.update_count = 0
            with torch.no_grad():
                self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
                self.half_weight_mirrored.copy_(self.half_weight)
                pass
            pass

        head1:torch.Tensor = torch.nn.functional.linear(input + 0.5, self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input - 0.5, self.half_weight_mirrored, self.bias)
        
        # Basically the only difference from the previous one.
        output = self.gramo(head1+head2)
        return output

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def convert_to_plain_Linear(self)->torch.nn.Linear:
        has_bias = bool(self.bias)
        result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
        with torch.no_grad():
            result.weight = self.half_weight+self.half_weight_mirrored
            result.bias.copy_(self.bias)
            pass
        return result
    
    def copy_from_plain_Linear(self, intput_linear_layer:torch.nn.Linear)->None:
        with torch.no_grad():
            self.half_weight = intput_linear_layer.weight/2.
            self.half_weight_mirrored.copy_(self.half_weight)
            if intput_linear_layer.bias:
                self.bias.copy_(intput_linear_layer.bias)
                pass
            pass

    pass




# Known issue. I assume the bias is always true. So, not safety check for bias == false.

# class test_multi_MIG__relu(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mig1  = MirrorWithGramo(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, 0.5,\
#                                       *args, **kwargs)
#         self.mig2  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig3  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig4  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig5  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig6  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig7  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig8  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig9  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig10 = MirrorWithGramo(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, 0.01 \
#                                      , *args, **kwargs)

#         self.biases:List[torch.nn.Parameter] = []
       
#         self.biases.append(self.mig1.bias)
#         self.biases.append(self.mig2.bias)
#         self.biases.append(self.mig3.bias)
#         self.biases.append(self.mig4.bias)
#         self.biases.append(self.mig5.bias)
#         self.biases.append(self.mig6.bias)
#         self.biases.append(self.mig7.bias)
#         self.biases.append(self.mig8.bias)
#         self.biases.append(self.mig9.bias)
#         self.biases.append(self.mig10.bias)
#         self.bias_scale_factor :List[float] = [1. for i in range(10)]
#         self.bias_scale_factor[1-1] = 1.
#         self.bias_scale_factor[2-1] = 1.
#         self.bias_scale_factor[3-1] = 1.
#         self.bias_scale_factor[4-1] = 1.
#         self.bias_scale_factor[5-1] = 1.
#         self.bias_scale_factor[6-1] = 1.
#         self.bias_scale_factor[7-1] = 1.
#         self.bias_scale_factor[8-1] = 1.
#         self.bias_scale_factor[9-1] = 1.
#         self.bias_scale_factor[10-1] = 3.
#         #0.02 works for 3 to 6 layers. But when I tested 7 layers, it's very slow.

#         if True:
#             self.act1  = torch.nn.ReLU()
#             self.act2  = torch.nn.ReLU()
#             self.act3  = torch.nn.ReLU()
#             self.act4  = torch.nn.ReLU()
#             self.act5  = torch.nn.ReLU()
#             self.act6  = torch.nn.ReLU()
#             self.act7  = torch.nn.ReLU()
#             self.act8  = torch.nn.ReLU()
#             self.act9  = torch.nn.ReLU()
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number :float = 1.
#         x = self.mig1 (x) 
#         x = self.act1(x)
#         x = x*a_small_number
#         x = self.mig2 (x) 
#         x = self.act2(x)
#         x = x*a_small_number
#         x = self.mig3 (x) 
#         x = self.act3(x)
#         x = x*a_small_number
#         x = self.mig4 (x) 
#         x = self.act4(x)
#         x = x*a_small_number
#         x = self.mig5 (x) 
#         x = self.act5(x)
#         x = x*a_small_number
#         x = self.mig6 (x) 
#         x = self.act6(x)
#         x = x*a_small_number
#         x = self.mig7 (x) 
#         x = self.act7(x)
#         x = x*a_small_number
#         x = self.mig8 (x) 
#         x = self.act8(x)
#         x = x*a_small_number
#         x = self.mig9 (x) 
#         x = self.act9(x)
#         x = x*a_small_number
#         return self.mig10(x) #move this line up and down. Each time 3 lines.
    
#     def before_optim_step(self)->None:
#         for i in range(len(self.biases)):
#             bias_param:torch.nn.Parameter = self.biases[i]
#             if bias_param is not None:
#                 if bias_param.grad is not None:
#                     bias_param.grad *= self.bias_scale_factor[i]



# iter_per_print = 1000#500
# print_count = 333#5
# # model = model.cuda()
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     model.before_optim_step()# This line is important.
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(epoch+1, "loss", loss)
#       
# print("pred  ", pred.T)

# print("target", target.T)

# jkldfs=345789











