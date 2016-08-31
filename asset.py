# (2016) Developed by Cheyenne Biolsi for Bayesquare Foundation Inc.

from arch.univariate import arch_model
import numpy as np

class Asset:
    
    def __init__(self, name, assethistory = [], classvalues = [], positivecount = 0,
                 positivetotal = 0.0, negativecount = 0, negativetotal = 0.0,
                 lastvolatility = 0):
        self.name = name
        self.assethistory = assethistory
        self.classvalues = classvalues
        self.positivecount = positivecount
        self.positivetotal = positivetotal
        self.negativecount = negativecount
        self.negativetotal = negativetotal
        self.lastvolatility = lastvolatility
    
    def recount_averages(self):
        increase = 0.0; increaseCount = 0.0
        decrease = 0.0; decreaseCount = 0.0
        for row in self.assethistory:
            if row[0]>0:
                increase+=row[0]
                increaseCount+=1
            else:
                decrease+=row[0]
                decreaseCount+=1
        self.positivecount = increaseCount
        self.positivetotal = increase
        self.negativecount = decreaseCount
        self.negativetotal = decrease
    
    def averages(self):
        return self.positivetotal/self.positivecount, self.negativetotal/self.negativecount
    
    def update(self, data, classValue):
        self.assethistory = np.concatenate((self.assethistory, data), axis = 0)
        self.classvalues = np.append(self.classvalues, classValue)
        if data[0,0] > 0:
            self.positivecount+=1
            self.positivetotal+=data[0,0]
        else:
            self.negativecount+=1
            self.negativetotal+=data[0,0]
        if self.assethistory[0,0] > 0:
            self.positivecount-=1
            self.positivetotal-=self.assethistory[0,0]
        else:
            self.negativecount-=1
            self.negativetotal-=self.assethistory[0,0]
        self.assethistory = self.assethistory[1:,:]
        self.classvalues = self.classvalues[1:]
    
    def update_volatility(self):
        returns = self.assethistory[:, 0].tolist()
        am = arch_model(returns)
        res=am.fit(disp='off')
        #print(res.summary())
        omega = res.params[1]
        alpha = res.params[2]
        beta = res.params[3]
        if self.lastvolatility == 0:
            prevVol = np.array(returns).var()
        else:
            prevVol = self.lastvolatility
        sigma2 = omega + alpha*(returns[-1]**2) + beta*prevVol
        self.lastvolatility = sigma2
        return(self.lastvolatility)