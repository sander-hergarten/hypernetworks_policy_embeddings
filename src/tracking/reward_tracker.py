import numpy as np 

class MeanTracker:
    episodes = []

    def reset(self, **environment_interface):
        self.episodes.append([])

    def __call__(self, step_data, **environment_interface):
        self.episodes[-1].append(step_data[1]) 

    def log(self):
        return np.mean(self.episodes, axis=0)
        

    
