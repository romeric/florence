import numpy as np
import matplotlib.pyplot as plt

lini = [2.089056015010000 ,  2.282796144490000 ,  2.264978170390000 ,  2.089366912840000,   2.086667060850000]
lin = [2.010993957520000,   2.112376928330000,   2.181015968320000,   2.023082017900000,   2.146771907810000]
nonlin = [6.205855131150000,   5.649722099300000,   5.465250015260000,   5.583621025090000,   5.693521976470000]

a= np.mean(lin)
b= np.mean(lini)
c= np.mean(nonlin)

out = [a/a,b/a,c/a]
# print 

ll = len(lini)
print ll
plt.bar(ll,out,width=1.,alpha=0.4)
plt.xlabel(r'$Elements$',fontsize=18)
plt.ylabel(r'$Scaled\, Jacobian$',fontsize=18)