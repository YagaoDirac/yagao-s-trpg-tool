from typing import Optional
from copy import deepcopy

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _float_list_equal(list_1:list[float], list_2:list[float], epsilon=0.001)->bool:
    assert list_1.__len__() == list_2.__len__()
    assert epsilon>0.
    for ii in range(list_1.__len__()):
        from_list_1 = list_1[ii]
        from_list_2 = list_2[ii]
        if abs(from_list_1-from_list_2)>epsilon:
            return False
        pass#for
    return True
if "test" and __DEBUG_ME__() and True:
    assert _float_list_equal([1., 2, 3], [1., 2, 3])
    assert _float_list_equal([1., 2, 3], [1.0009, 2, 3])
    assert not _float_list_equal([1., 2, 3], [1.0011, 2, 3])
    pass
        


class Consequence_raw():
    to_what:str
    how_much_I_provide:float
    how_much_Im_affected:float
    _index_mem:int
    def __init__(self, to_what:str, how_much_I_provide:float, how_much_Im_affected = 0.):
        assert to_what.__len__()>0
        assert how_much_I_provide>0.
        assert how_much_I_provide<1.
        assert how_much_Im_affected>=0.
        assert how_much_Im_affected<1.
        
        self.to_what = to_what
        self.how_much_I_provide = how_much_I_provide
        self.how_much_Im_affected = how_much_Im_affected
        self._index_mem = -1
        pass#/function
    pass#/class

class Content_item_raw():
    '''
    ???
    step1, create this object, add consequence.
    step2, process this in **class contest**.
    '''
    name:str
    #keywords:list[str]#以后搜索的时候用的。
    content:str
    threshold:float
    raw_consequence_list:list[Consequence_raw]
    
    needs_print:bool
    #                      trans_strength   name   the object.                                                                             
    #strength:float 只有这个是动的
    def __init__(self, name:str, keywords:list[str],\
                    content:str, threshold:float, needs_print = True):
        assert threshold>0.
        assert threshold<1.
        self.name = name
        #self.keywords = deepcopy(keywords)
        self.content = content
        self.threshold = threshold
        self.raw_consequence_list = []
        self.needs_print = needs_print
        pass
    def add_consequence(self, csq_raw:Consequence_raw):
        for self_csq_raw in self.raw_consequence_list:
            assert self_csq_raw.to_what != csq_raw.to_what, "Already added."
            pass
        self.raw_consequence_list.append(csq_raw)
        pass#/function
    def add_consequence_list(self, consequence_list:list[Consequence_raw]):
        for input_item in consequence_list:
            self.add_consequence(input_item)
            pass
        pass#/function
    
    def __str__(self)->str:
        if not self.needs_print:
            return ""
        return f"{self.name}:{self.content}"
    
    def __repr__(self)->str:
        result = f"{self.name}: {self.content}."
        if not self.needs_print:
            result += "(hidden)"
            pass
        return result
    pass#</class



class Consequence():
    "designed to be used in **class content_item**"
    to_index:int
    how_much_I_provide:float
    #how_much_Im_affected:float
    def __init__(self, to_index:int, based_on:Consequence_raw):
        assert to_index>=0
        self.to_index = to_index
        self.how_much_I_provide = based_on.how_much_I_provide
        #self.how_much_Im_affected = based_on.how_much_Im_affected
        pass#/function
    pass#/class
    
class Content_item():
    '''
    step1, create this object, add consequence.
    step2, use this.
    '''
    index_in_list:int
    name:str
    #keywords:list[str]#以后搜索的时候用的。
    content:str
    threshold:float
    flushed_consequence_list:list[Consequence]
    
    is_sorted:bool
    needs_print:bool
    #                      trans_strength   name   the object.                                                                             
    #strength:float 只有这个是动的
    def __init__(self, index:int, what:Content_item_raw):
        self.index_in_list = index
        self.name = what.name
        #self.keywords = what.keywords
        self.content = what.content
        self.threshold = what.threshold
        self.flushed_consequence_list = []
        self.needs_print = what.needs_print
        pass
    
    def _sort(self, content_dot_data:list['Content_item']):
        "call this from class content."
        
        self.flushed_consequence_list.sort(reverse= True, key=lambda csq:csq.how_much_I_provide/content_dot_data[csq.to_index].threshold)
        for flushed_consequence in self.flushed_consequence_list:
            assert flushed_consequence.how_much_I_provide>=content_dot_data[flushed_consequence.to_index].threshold,\
                f"self({self.name}) has a consequence item of ({content_dot_data[flushed_consequence.to_index].name\
                    }), the trans_strength is too small, this transition will never happen."
            
        self.is_sorted = True
        pass
    
    def __str__(self)->str:
        if not self.needs_print:
            return ""
        return f"{self.name}:{self.content}"
    
    def __repr__(self)->str:
        result = f"{self.index_in_list}#{self.name}. "
        if not self.needs_print:
            result += "(hidden)"
            pass
        return result
    
    def get_consequence(self, strength:float, content_dot_data:list['Content_item'])->list[tuple[float, 'Content_item']]:
        #                                                                                strength^^^^^
        "return [(real strength, what)]"
        assert self.is_sorted
        result:list[tuple[float, Content_item]] = []
        for csq in self.flushed_consequence_list:
            result_strength = strength * csq.how_much_I_provide
            if result_strength>=content_dot_data[csq.to_index].threshold:
                result.append((result_strength, content_dot_data[csq.to_index]))
                pass
            else:
                break# The list is sorted. From here on, nothing will happen.
            pass#/for to_where
        return result#/function
    
    pass#</class

class Content():
    data:list[Content_item]
    name_to_index_map:dict[str,int]
    
    if False:
        @staticmethod
        def __load_from_file(filename:str)->'Content':
            assert False, "还没写，不急。"
            
            def find_next(line:str, start_index:int)->tuple[str,int]:
                end_pos = start_index
                while True:
                    end_pos = the_line.find(" ", start_index)
                    if end_pos == start_index:
                        #still nothing useful, continue searching.
                        start_index+=1
                        pass
                    elif -1 == end_pos:
                        #nothing left, return empty string.
                        return "", -1
                    else:
                        #found something, return it.
                        return line[start_index, end_pos], end_pos
                    #notail
                    pass#while
                pass#function.
            
            the_item_dict:dict[str,str]
            #the_keyword_dict:map[str,list[str]]
            the_consequence_dict:dict[str,list[Consequence_raw]]
            with open(filename, 'r', encoding="utf-8") as file:
                while True:
                    the_line = file.readline()
                    if "" == the_line:#eof
                        break
                    
                    ci_name, _pos = find_next(the_line, 0)
                    other_name, _pos = find_next(the_line, _pos)
                    next_item, _pos= find_next(the_line, _pos)
                    do_it_again = False
                    try:
                        how_much_I_provide = float(next_item)
                        pass
                    except ValueError as e:
                        do_it_again = True
                        pass
                    if do_it_again:
                        next_item, _pos= find_next(the_line, _pos)
                        how_much_I_provide = float(next_item)
                        pass
                    
                    next_item, _pos= find_next(the_line, _pos)
                    if next_item != "":
                        how_much_Im_affected = float(next_item)
                        pass
                    else:
                        how_much_Im_affected = 0.
                        pass
                    
                    the_consequence_dict
                    
                    
                    #tail
                    pass#while
                pass#with file
        pass    
    
    def __init__(self, input:list[Content_item_raw]):
        self.data = []
        self.name_to_index_map:dict[str,int] = {}
        
        #<build up the index>
        for index in range(input.__len__()):
            item_raw = input[index]
            self.data.append(Content_item(index,item_raw))
            assert self.name_to_index_map.get(item_raw.name) is None, f'Name({item_raw.name}) duplicated.'
            self.name_to_index_map[item_raw.name] = index
            pass
        #</build up the index>
        assert self.data.__len__() == input.__len__()
        assert self.name_to_index_map.__len__() == input.__len__()
        #<the forward affect.
        for index in range(input.__len__()):
            ci = self.data[index]
            ci_raw = input[index]
            for csq_raw in ci_raw.raw_consequence_list:
                to_index = self.name_to_index_map.get(csq_raw.to_what)
                assert isinstance(to_index, int), "the name was not found previously."
                assert csq_raw is not None, f"{self.data[index].name} has a consequence named {csq_raw.to_what}, {None\
                                    }this name doesn't exist. Search ***{self.data[index].name}*** and correct the{None\
                                    } consequence item of ***{csq_raw.to_what}*** to "
                ci.flushed_consequence_list.append(Consequence(to_index, csq_raw))
                if csq_raw.how_much_Im_affected>0.:
                    csq_raw._index_mem = to_index
                    pass
                pass
            pass
        #</the forward affect.
        #<the backward affect.
        for other_index in range(input.__len__()):
            other_ci = self.data[other_index]
            other_ci_raw = input[other_index]
            for other_csq_raw in other_ci_raw.raw_consequence_list:
                if other_csq_raw.how_much_Im_affected>0.:
                    ci = self.data[other_csq_raw._index_mem]
                    _already_exists = False
                    for csq in ci.flushed_consequence_list:
                        if csq.to_index == other_ci.index_in_list:
                            _already_exists = True
                            # this consequence already exists. It should be some error.
                            if csq.how_much_I_provide == 0.:
                                # it's still weird, but acceptable.
                                ci.flushed_consequence_list.append(Consequence(other_index, other_csq_raw))
                                pass
                            else:
                                assert csq.how_much_I_provide == other_csq_raw.how_much_Im_affected, f"{ci.name\
                                                } provides affect to {other_ci.name}, while {other_ci.name \
                                                } accepts affect from {ci.name} WITH A DIFFERENT VALUE.{None\
                                                } They must equal, or one of them must be 0."
                                #check and do nothing.
                                pass
                            pass#csq.to_index == other_ci.index_in_list:
                        pass#for csq in ci
                    if not _already_exists:
                        _temp_csq = Consequence(other_index, other_csq_raw)
                        _temp_csq.how_much_I_provide = other_csq_raw.how_much_Im_affected
                        ci.flushed_consequence_list.append(_temp_csq)
                        pass
                    pass#if other_csq_raw.how_much_Im_affected>0.:
                pass#for other_csq_raw in other_ci_raw
            pass#for other_index in range
        #</the backward affect.
        
        #<sort
        for ci in self.data:
            ci._sort(self.data)                
            pass
        #</sort
        
        pass#/function
    
if "test" and __DEBUG_ME__() and True:
    ci_raw_list = []
    ci_raw = Content_item_raw(name="事件A", keywords=[], content="事件A发生了。", threshold=0.3)
    ci_raw.add_consequence(Consequence_raw("事件B", 0.8, 0.5))
    ci_raw_list.append(ci_raw)
    ci_raw = Content_item_raw(name="事件B", keywords=[], content="事件B发生了。", threshold=0.1)
    ci_raw_list.append(ci_raw)
    
    ct = Content(ci_raw_list)
    assert ct.data.__len__() == 2
    assert ct.data[0].name == "事件A"
    assert ct.data[0].threshold == 0.3
    assert ct.data[0].flushed_consequence_list.__len__() == 1
    assert ct.data[0].flushed_consequence_list[0].how_much_I_provide == 0.8
    assert ct.data[0].flushed_consequence_list[0].to_index == 1
    
    assert ct.data[1].name == "事件B"
    assert ct.data[1].threshold == 0.1
    assert ct.data[1].flushed_consequence_list.__len__() == 1
    assert ct.data[1].flushed_consequence_list[0].how_much_I_provide == 0.5
    assert ct.data[1].flushed_consequence_list[0].to_index == 0
    
    
    
    
    ci_raw_list = []
    ci_raw = Content_item_raw(name="排序主体", keywords=[], content="", threshold=0.3)
    ci_raw.add_consequence(Consequence_raw("a6_12", 0.6))
    ci_raw.add_consequence(Consequence_raw("a6_10", 0.6))
    ci_raw.add_consequence(Consequence_raw("a9_12", 0.9))
    ci_raw.add_consequence(Consequence_raw("a9_10", 0.9))
    ci_raw_list.append(ci_raw)
    ci_raw_list.append(Content_item_raw(name="a6_12", keywords=[], content="", threshold=0.12))
    ci_raw_list.append(Content_item_raw(name="a6_10", keywords=[], content="", threshold=0.10))
    ci_raw_list.append(Content_item_raw(name="a9_12", keywords=[], content="", threshold=0.12))
    ci_raw_list.append(Content_item_raw(name="a9_10", keywords=[], content="", threshold=0.10))
    
    ct = Content(ci_raw_list)
    assert ct.data.__len__() == 5
    assert ct.data[0].name == "排序主体"
    assert ct.data[0].threshold == 0.3
    assert ct.data[0].flushed_consequence_list.__len__() == 4
    assert ct.data[0].flushed_consequence_list[0].to_index == 4
    assert ct.data[0].flushed_consequence_list[1].to_index == 3
    assert ct.data[0].flushed_consequence_list[2].to_index == 2
    assert ct.data[0].flushed_consequence_list[3].to_index == 1
    
    assert ct.data[1].name == "a6_12"
    assert ct.data[2].name == "a6_10"
    assert ct.data[3].name == "a9_12"
    assert ct.data[4].name == "a9_10"
    
    pass




class Scenario():
    strength:list[float]
    CONTENT:Content
    def __init__(self, content:Content):
        self.CONTENT = content
        self.strength = [0.]*content.data.__len__()
        pass#/function
    if False:
        # #<write the index into items  and  flush
        # for ii in range(CONTENT.__len__()):
        #     item = CONTENT[ii]
        #     item.index_in_list = ii
        #     pass
        # for ci in CONTENT:
        #     for ii in range(ci.raw_consequence_list.__len__()):
        #         csq = ci.raw_consequence_list[ii]
        #         #find the real object with the same name
        #         found = False
        #         for jj in range(CONTENT.__len__()):
        #             obj = CONTENT[jj]
        #             if obj.name == csq[1]:
        #                 found = True
        #                 ci.raw_consequence_list[ii] = (csq[0],csq[1],obj)
        #                 break
        #             pass#/for jj
        #         assert found, f"{ci.name}'s {ii}th consequence is assigned with a wrong name."
        #         pass#/for ii
        #     ci.flush()
        #     pass#/for ci
        # pass
        # #</write the index into items  and  flush
        # pass
        pass
    
    def _name_or_index_to_index(self, what:int|str)->int:
        if isinstance(what, int):
            index = what
            pass
        elif isinstance(what, str):
            temp_result = self.CONTENT.name_to_index_map.get(what)
            assert isinstance(temp_result, int), "Name not found."
            index = temp_result
            pass
        else:
            assert False,"unreachable"
            pass
        return index
    
    def add(self, what:int|str, strength:float):
        assert strength>0.
        self._update(what, strength, force=False)
        pass
    
    def update(self, what:int|str, strength:float):
        assert strength>0.
        self._update(what, strength, force=True)
        pass
        
    def clear(self, what:int|str):
        self._update(what, strength = 0., force=True)
        pass
        
    def _update(self, what:int|str, strength:float, force:bool):
        assert strength<1.
        index = self._name_or_index_to_index(what)
        if not force:
            assert self.strength[index] ==0.,"Already added. Can NOT add again."
            pass
        self.strength[index] = strength
        pass
    
    def clear_all(self):
        self.strength = [0.]*(self.strength.__len__())
        pass
    
    def get(self,what:int|str)->float:
        index = self._name_or_index_to_index(what)
        return self.strength[index]
        pass
    
    
    
    
    def calc(self)->list[float]:
        check_list:list[tuple[float, Content_item]] = []
        #             strength^^^^^   
        
        #<init before loop
        for index in range(self.strength.__len__()):
            strength = self.strength[index]
            ci = self.CONTENT.data[index]
            if strength>=ci.threshold:
                check_list.append((strength, ci))
                pass
            pass#/for ii
        #</init before loop
        
        #<init result and add existing items to it>
        result_as_strength:list[float] = [0]*self.strength.__len__()
        for strength, item in check_list:
            result_as_strength[item.index_in_list] = strength
            pass
        #</init result and add existing items to it>
        
        while check_list.__len__()>0:
            (the_strength, the_item) = check_list.pop()
            candidates:list[tuple[float, Content_item]] = the_item.get_consequence(the_strength, self.CONTENT.data)
            #too weak case are already filtered out by the function uppon.
            for candidate_strength, candidate_what in candidates:
                # old assert isinstance(candidate_what, Content_item)
                # index_in_list = candidate_what.index_in_list
                # #<add new useful element into check list
                # # if already_checked_as_strength[index_in_list]<cancidate_strength:
                # #     already_checked_as_strength[index_in_list] = cancidate_strength
                # #     check_list.append(candidate)
                # #     pass
                # #</add new useful element into check list
                #<if it can overwrite on result
                if result_as_strength[candidate_what.index_in_list] < candidate_strength:
                    result_as_strength[candidate_what.index_in_list] = candidate_strength
                    check_list.append((candidate_strength, candidate_what))
                    pass
                #</if it can overwrite on result
            #notail
            pass
        
        return result_as_strength
            
    # @staticmethod
    # def convert_calc_result_to_report(calc_result:list[float], )->str:
    #     for strength in calc_result:
    #         assert False,' 继续。'
        
    pass#/class

if "test" and __DEBUG_ME__() and True:
    #basic.
    ci_raw_list = []
    ci_raw = Content_item_raw(name="事件A", keywords=[], content="事件A发生了。", threshold=0.3)
    ci_raw.add_consequence(Consequence_raw("事件B", 0.8, 0.5))
    ci_raw_list.append(ci_raw)
    ci_raw = Content_item_raw(name="事件B", keywords=[], content="事件B发生了。", threshold=0.1)
    ci_raw_list.append(ci_raw)
    
    ct = Content(ci_raw_list)
    sc = Scenario(ct)
    sc.add("事件A",0.654)
    assert sc.strength == [0.654, 0]
    sc.add(1, 0.765)
    assert sc.strength == [0.654, 0.765]
    
    
    
    #A to all.
    ci_raw_list = []
    ci_raw = Content_item_raw(name="排序主体", keywords=[], content="", threshold=0.001)
    ci_raw.add_consequence(Consequence_raw("a6_12", 0.6))
    ci_raw.add_consequence(Consequence_raw("a6_10", 0.6))
    ci_raw.add_consequence(Consequence_raw("a9_12", 0.9))
    ci_raw.add_consequence(Consequence_raw("a9_10", 0.9))
    ci_raw_list.append(ci_raw)
    ci_raw_list.append(Content_item_raw(name="a6_12", keywords=[], content="", threshold=0.12))
    ci_raw_list.append(Content_item_raw(name="a6_10", keywords=[], content="", threshold=0.10))
    ci_raw_list.append(Content_item_raw(name="a9_12", keywords=[], content="", threshold=0.12))
    ci_raw_list.append(Content_item_raw(name="a9_10", keywords=[], content="", threshold=0.10))
    
    
    ct = Content(ci_raw_list)
    sc = Scenario(ct)    
    sc.add(0, 0.5)
    result = sc.calc()
    assert _float_list_equal(result, [0.5,   0.3,0.3,0.45,0.45])
    sc.update(0, 0.19)
    result = sc.calc()
    assert _float_list_equal(result, [0.19,   0.,   0.114,0.171,0.171])
    sc.update(0, 0.15)
    result = sc.calc()
    assert _float_list_equal(result, [0.15,   0.,0.,   0.135,0.135])
    sc.update(0, 0.13)
    result = sc.calc()
    assert _float_list_equal(result, [0.13,   0.,0.,0.,   0.117])
    sc.update(0, 0.1)
    result = sc.calc()
    assert _float_list_equal(result, [0.1,   0.,0.,0.,0.])
    assert sc.CONTENT.data[0].name == "排序主体"
    assert sc.CONTENT.data[0].threshold <0.1
    
    
    #ciclic, A->B->C->A
    ci_raw_list = []
    ci_raw = Content_item_raw(name="A", keywords=[], content="", threshold=0.01)
    ci_raw.add_consequence(Consequence_raw("B", 0.5))
    ci_raw_list.append(ci_raw)
    ci_raw = Content_item_raw(name="B", keywords=[], content="", threshold=0.01)
    ci_raw.add_consequence(Consequence_raw("C", 0.5))
    ci_raw_list.append(ci_raw)
    ci_raw = Content_item_raw(name="C", keywords=[], content="", threshold=0.01)
    ci_raw.add_consequence(Consequence_raw("A", 0.5))
    ci_raw_list.append(ci_raw)
    
    ct = Content(ci_raw_list)
    sc = Scenario(ct)    
    sc.add(0, 0.8)
    result = sc.calc()
    assert _float_list_equal(result, [0.8, 0.4, 0.2])
    
    
    #mutually
    ci_raw_list = []
    ci_raw = Content_item_raw(name="A", keywords=[], content="", threshold=0.01)
    ci_raw.add_consequence(Consequence_raw("B", 0.5, 0.5))
    ci_raw_list.append(ci_raw)
    ci_raw = Content_item_raw(name="B", keywords=[], content="", threshold=0.01)
    ci_raw_list.append(ci_raw)
    
    ct = Content(ci_raw_list)
    sc = Scenario(ct)    
    sc.add(0, 0.8)
    result = sc.calc()
    assert _float_list_equal(result, [0.8, 0.4])
    sc.clear_all()
    sc.update(1, 0.8)
    result = sc.calc()
    assert _float_list_equal(result, [0.4, 0.8])
    
    
    pass
    
    
    
    
    