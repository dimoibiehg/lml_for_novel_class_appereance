from mape.mape import MAPE

class LifelongSelfAdaptation:
    def __init__(self, mape:MAPE):
        self.mape:MAPE = mape
    
    def start(self):
        for i in tqdm(range(1, 349)):
            if(i%20 == 0):
                pass
            self.mape.monitor_and_analyse()