import numpy as np
import os
from warnings import warn


def WitherdenQuadraturePointsHex(C):

    path = os.path.dirname(os.path.realpath(__file__))
    path += '/Tables/hex/'
    C +=1

    if C<3:
        d = 3
    if C==3:
        d = 5
    elif C==4:
        d = 7
    elif C==5:
        d = 9
    elif C==6:
        d = 11
    else:
        # warn("Witherden points beyond C=4 not available")
        raise ValueError("Witherden points beyond C=4 not available")

    for i in os.listdir(path):
        if 'witherden-vincent-n' in i:
            if 'd'+str(d) in i:
                zw = np.loadtxt(path+i)
                break


    return zw