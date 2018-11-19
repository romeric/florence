import numpy as np

def MeanValueCoordinateMapping(dpoint, uv, physical_points):
    """MVC mapping from parametric uv to physical 3D
        inputs:
            dpoint:             [list, tuple or 1D array of floats] desired uv point
            uv:                 [2D array] of parametric uv points of polygon vertices
            physical_points:    [2D array] of physical points of polygon vertices
    """

    norm = np.linalg.norm

    counter = 0
    cp = False
    for i, p in enumerate(uv):
        if np.abs(norm(p-dpoint)) < 1e-9:
            counter = i
            cp = True
            break
    if cp:
        return physical_points[counter,:]


    dpoint_tile = np.tile(dpoint,uv.shape[0]).reshape(uv.shape[0],uv.shape[1])

    segments = dpoint_tile - uv
    seg_lengths = norm(segments, axis=1)
    num_vertices = uv.shape[0]

    alphas = []
    for i in range(num_vertices):
        if i<num_vertices-1:
            n0 = norm(segments[i,:])
            n1 = norm(segments[i+1,:])
            s0 = norm(segments[i,:]/n0 + segments[i+1,:]/n1)
            s1 = norm(segments[i,:]/n0 - segments[i+1,:]/n1)
            a  = 2.*np.arctan2(s1,s0)
        else:
            n0 = norm(segments[i,:])
            n1 = norm(segments[0,:])
            s0 = norm(segments[i,:]/n0 + segments[0,:]/n1)
            s1 = norm(segments[i,:]/n0 - segments[0,:]/n1)
            a  = 2.*np.arctan2(s1,s0)
        alphas.append(a)


    ws = []
    for i in range(num_vertices):
        if i==0:
            a0 = alphas[-1]
            a1 = alphas[i]
            n1 = seg_lengths[i]
            w = (np.tan(a0/2.) + np.tan(a1/2.))/n1
        else:
            a0 = alphas[i-1]
            a1 = alphas[i]
            n1 = seg_lengths[i]
            w = (np.tan(a0/2.) + np.tan(a1/2.))/n1
        ws.append(w)

    ws = np.array(ws)

    lmbs = ws / np.sum(ws)

    candidate_point = np.zeros((physical_points.shape[1]))
    for i in range(num_vertices):
        candidate_point += physical_points[i,:]*lmbs[i]

    candidate_point = candidate_point.reshape(candidate_point.shape[0],1)
    candidate_point = candidate_point.ravel()

    return candidate_point