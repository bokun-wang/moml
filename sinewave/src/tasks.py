import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax
        
    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        
        return self.amplitude * np.sin(self.phase + x)
        
    def sample_data(self, size=1):
        """
        Sample data from this task.
        
        returns: 
            x: the feature vector of length size
            y: the target vector of length size
        """
        
        x = np.random.uniform(self.xmin, self.xmax, size)
        y = self.true_function(x)
        
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        
        return x, y

    def sample_test_data(self, size=1, seed=2021):
        """
        Sample data from this task.

        returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        seed_everything(seed)
        x = np.random.uniform(self.xmin, self.xmax, size)
        y = self.true_function(x)

        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

        return x, y

class Sine_Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """
    
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max
        
    def sample_task(self):
        """
        Sample from the task distribution.
        
        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        return Sine_Task(amplitude, phase, self.x_min, self.x_max)

def sample_amp_pha(amp_min, amp_max, pha_min, pha_max, seed=2021):
    seed_everything(seed)
    amplitude = np.random.uniform(amp_min, amp_max)
    phase = np.random.uniform(pha_min, pha_max)
    return amplitude, phase