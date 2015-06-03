import numpy as np
import numpy.linalg as la
def NormalDistance(p1, p2, p3, p4):
	# """ This function returns normal distance between two parallel lines. 
	    # The function is from Dan Sunday's Geometry Algorithms originally written in C++ """
    u = p1 - p2;
    v = p3 - p4;
    w = p2 - p4;
    
    a = np.dot(u,u);
    b = np.dot(u,v);
    c = np.dot(v,v);
    d = np.dot(u,w);
    e = np.dot(v,w);
    D = a*c - b*b;
    sD = D;
    tD = D;
    
    SMALL_NUM = 0.00000001;
    
     # compute the line parameters of the two closest points
    if (D < SMALL_NUM):  # the lines are almost parallel
        sN = 0.0;       # force using point P0 on segment S1
        sD = 1.0;       # to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    else:                # get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0):   # sc < 0 => the s=0 edge is visible       
            sN = 0.0;
            tN = e;
            tD = c;
        elif (sN > sD): # sc > 1 => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
    
    if (tN < 0.0):            # tc < 0 => the t=0 edge is visible
        tN = 0.0;
        # recompute sc for this edge
        if (-d < 0.0):
            sN = 0.0
        elif (-d > a):
            sN = sD
        else:
            sN = -d
            sD = a
    elif (tN > tD):       # tc > 1 => the t=1 edge is visible
        tN = tD;
        # recompute sc for this edge
        if ((-d + b) < 0.0):
            sN = 0
        elif ((-d + b) > a):
            sN = sD
        else: 
            sN = (-d + b)
            sD = a
    
    # finally do the division to get sc and tc
    if(abs(sN) < SMALL_NUM):
        sc = 0.0
    else:
        sc = sN / sD;
    
    if(abs(tN) < SMALL_NUM):
        tc = 0.0
    else:
        tc = tN / tD
    
    # get the difference of the two closest points
    dP = w + (sc * u) - (tc * v)  # = S1(sc) - S2(tc)

    return la.norm(dP)