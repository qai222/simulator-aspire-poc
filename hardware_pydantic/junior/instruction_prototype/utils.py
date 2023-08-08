from functools import wraps

from hardware_pydantic.junior.junior_lab import JuniorInstruction


def ins_list_path_graph(func):
    @wraps(func)
    def ins_list(*args, **kwargs):
        lst = func(*args, **kwargs)
        JuniorInstruction.path_graph(lst)
        return lst

    return ins_list


def chain_ins_lol(lol: list[list[JuniorInstruction]]):
    for i in range(len(lol) - 1):
        former = lol[i]
        latter = lol[i + 1]
        latter[0].preceding_instructions.append(former[-1].identifier)


def ins_diverge_or_converge(ins1: JuniorInstruction, ins_lst: list[JuniorInstruction], diverge=True):
    for i in ins_lst:
        if diverge:
            i.preceding_instructions.append(ins1.identifier)
        else:
            ins1.preceding_instructions.append(i.identifier)
