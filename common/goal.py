from enum import Enum
from strenum import StrEnum
from abc import ABC
import math 
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from random import randint

class GoalType(Enum):
    THRESHOLD = 1
    OPTIMIZATION = 2

class CompareType(Enum):
    GREATER = 1
    LESS = 2

class OptimizationType(Enum):
    MAX = 1
    MIN = 2
class TargetType(Enum):
    def __str__(self):
        return self.name
    LATENCY = 1
    PACKETLOSS = 2
    ENERGY_CONSUMPTION = 3


class TargetName(StrEnum):
    LATENCY = "latency"
    PACKETLOSS = "packetloss"
    ENERGY_CONSUMPTION = "energyconsumption"


class TargetRange(Enum):
    LATENCY = [0,100]
    PACKETLOSS = [0,100]
    ENERGY_CONSUMPTION = [0, math.inf]

class Goal(ABC):
    def __init__(self, goal_type, target_type):
        self.type = goal_type
        self.target_type = target_type

class ThrehsholdGoal(Goal):
    def __init__(self, target_type, compare_type, value):
        self.compare_type = compare_type
        self.value = value
        super().__init__(GoalType.THRESHOLD, target_type)

class OptimizationGoal(Goal):
    def __init__(self, target_type, optimization_type):
        self.optimization_type = optimization_type
        super().__init__(GoalType.OPTIMIZATION, target_type)


def utility_fucntion(target_type, value):
    if(target_type == TargetType.PACKETLOSS):
        return -0.01 * value + 1
        # a = 100.0
        # if((value <= a) and (value >= 0.0)):
        #     return (a - value)/(a*(value + 1.0))
        # return 0.0
        # if((value <= 15.0) and (value >= 0.0)):
        #     return 1.0
        # elif((value > 15.0) and (value <= 30.0)):
        #     return 0.5
        # # elif((value > 20.0) and (value <= 40.0)):
        # #     return 0.25
        # elif((value > 30.0) and (value <= 60.0)):
        #     return 0.125
        # elif(value > 60.0):
        #     return 0.0
        # else:
        #     raise Exception("Not in the range of packet loss")
        # a = 5.0
        # b = 30.0
        # x = 0.5
        # if((value <= a) and (value >= 0.0)):
        #     return (a + (a*x + x - 1.0) * value)/(a*(value + 1.0))
        # elif((value > a) and (value <= b)):
        #     return x
        # elif((value > b) and (value <= 100.0)):
        #     return (b * (100.0/value - 1.0) * x) / (100.0 - b)
        # else:
        #     raise Exception("Not in the range of packet loss")
        # if((value <= 5.0) and (value >= 0.0)):
        #     return (6.0/(value + 1.0) + 4.0) / 10.0
        # elif((value > 5.0) and (value <= 15.0)):
        #     return 0.5
        # elif((value > 15.0) and (value <= 100.0)):
        #     return (750.0/value - 7.5) / 85.0
        # else:
        #     raise Exception("Not in the range of packet loss")
    elif(target_type == TargetType.ENERGY_CONSUMPTION):
        if((value <= 13.0) and (value >= 0.0)):
            return 1.0
        # elif((value > 13.0) and (value <= 13.2)):
        #     return 0.5
        elif((value > 13.0) and (value <= 13.4)):
            return 0.25
        elif(value > 13.4):
            return 0
        else:
            raise Exception("Not in the range of energy consumtpion")

    else:
        raise Exception("TergetType other than packet loss and energy consumtpion has not been implemented.")
    
def weight_for_targte_type(target_type):
    if(target_type == TargetType.PACKETLOSS):
        return 0.8
    elif(target_type == TargetType.ENERGY_CONSUMPTION):
        return 0.2
    else:
        raise Exception("TergetType other than packet loss and energy consumtpion has not been implemented.")      
 
    
class DeltaIoTMultiObjective(BinaryProblem):
    def __init__(self, packet_loss: list, energy_consumption: list, latency: list):
        super(DeltaIoTMultiObjective, self).__init__()#reference_front=None
        self.packet_loss = packet_loss
        self.energy_consumption = energy_consumption
        self.latency = latency

        self.number_of_bits = len(self.latency)
        self.number_of_objectives = 3
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['PL', 'EC', 'LA']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        found_true = False
        for index, bits in enumerate(solution.variables[0]):
            if bits:
                solution.objectives[0] = self.packet_loss[index]
                solution.objectives[1] = self.energy_consumption[index]
                solution.objectives[2] = self.latency[index]
                found_true = True
                break
        if(~found_true):
            solution.objectives[0] = TargetRange.PACKETLOSS.value[1]
            solution.objectives[1] = TargetRange.ENERGY_CONSUMPTION.value[1]
            solution.objectives[2] = TargetRange.LATENCY.value[1]
        return solution
    
    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                    number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = [False] * self.number_of_bits
        new_solution.variables[0][randint(0, self.number_of_bits-1)] = True
            # [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return 'DeltaIoTv1'
    
    
    
    
class DeltaIoTTwoObjective(BinaryProblem):
    def __init__(self, objective1: list, objective2: list, max_objective1, max_objective2):
        super(DeltaIoTTwoObjective, self).__init__()#reference_front=None
        self.objective1 = objective1
        self.objective2 = objective2
        self.max_objective1 = max_objective1
        self.max_objective2 = max_objective2
        
        self.number_of_bits = len(self.objective1)
        self.number_of_objectives = 2
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['Obj1', 'Obj2']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        found_true = False
        for index, bits in enumerate(solution.variables[0]):
            if bits:
                solution.objectives[0] = self.objective1[index]
                solution.objectives[1] = self.objective2[index]
                found_true = True
                break
        if(~found_true):
            solution.objectives[0] = self.max_objective1
            solution.objectives[1] = self.max_objective2
        return solution
    
    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                    number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = [False] * self.number_of_bits
        new_solution.variables[0][randint(0, self.number_of_bits-1)] = True
            # [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return 'DeltaIoTv1'