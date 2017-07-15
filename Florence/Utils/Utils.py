from __future__ import print_function

# COMPARE STRINGS WHICH MIGHT CONTAIN UNICODES
############################################################################
def insensitive(string):
    """Given a string, returns its lower/upper case insensitive string"""
    if getattr(str,'casefold',None) is not None:
        insen = lambda str_name: str_name.casefold()
    else:
        insen = lambda str_name: str_name.upper().lower()

    return insen(string)


def par_unpickle(fem_solver, mesh, material, formulation, 
    Eulerx, Eulerp, NeumannForces, NodalForces, Residual, K,
    TotalDisp, velocities, accelrations):
    """unpickle tuple of objects"""

    import pickle, gc, time, os, shutil, errno
    from scipy.io import savemat

    # from tempfile import mkdtemp
    # tmp_dir = mkdtemp()
    
    tmp_dir = "/home/roman/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    fem_solver_file = os.path.join(tmp_dir,'fem_solver')
    mesh_file = os.path.join(tmp_dir,'mesh')
    material_file = os.path.join(tmp_dir,'material')
    Eulerx_file = os.path.join(tmp_dir,'Eulerx')
    Eulerp_file = os.path.join(tmp_dir,'Eulerp')

    t_unpickle = time.time()

    # main_data = MainData
    f = file(fem_solver_file, 'wb')
    pickle.dump(fem_solver,f,pickle.HIGHEST_PROTOCOL)
    f = file(mesh_file, 'wb')
    pickle.dump(mesh,f,pickle.HIGHEST_PROTOCOL)
    f = file(material_file, 'wb')
    pickle.dump(material,f,pickle.HIGHEST_PROTOCOL)
    f = file(Eulerx_file, 'wb')
    pickle.dump(Eulerx,f,pickle.HIGHEST_PROTOCOL)
    f = file(Eulerp_file, 'wb')
    pickle.dump(Eulerp,f,pickle.HIGHEST_PROTOCOL)
    
    savemat(os.path.join(tmp_dir,'rest.mat'),
        {'C':MainData.C,'ndim':MainData.ndim,'nvar':MainData.nvar},do_compression=True)
    del MainData, mesh, material, Eulerx, TotalPot, f
    gc.collect()

    print('Time taken for unpickling was', time.time() - t_unpickle, 'seconds')

    # try:
    #     # delete directory
    #     shutil.rmtree(tmp_dir)  
    # except OSError as exc:
    #     # ENOENT - no such file or directory
    #     if exc.errno != errno.ENOENT:  
    #         raise  IOError("No directory to delete")

    # f = file(main_data_file, 'rb')
    # MainData = pickle.load(f)
    # print MainData.solver

    return tmp_dir


def par_pickle(tmp_dir):

    import pickle, os, time, shutil, errno
    from scipy.io import loadmat

    t_pickle = time.time()

    # main_data_file = tmp_dir[1]
    # mesh_file =  tmp_dir[2]
    # material_file =  tmp_dir[3]
    # Eulerx_file =  tmp_dir[4]
    # TotalPot_file =  tmp_dir[5]

    main_data_file = os.path.join(tmp_dir,'main_data')
    mesh_file = os.path.join(tmp_dir,'mesh')
    material_file = os.path.join(tmp_dir,'material')
    Eulerx_file = os.path.join(tmp_dir,'Eulerx')
    TotalPot_file = os.path.join(tmp_dir,'TotalPot')

    f = file(main_data_file, 'rb')
    MainData = pickle.load(f)
    # print MainData.Domain
    f = file(mesh_file, 'rb')
    mesh = pickle.load(f)
    f = file(material_file, 'rb')
    material = pickle.load(f)
    f = file(Eulerx_file, 'rb')
    Eulerx = pickle.load(f)
    f = file(TotalPot_file, 'rb')
    TotalPot = pickle.load(f)

    # try:
    #     # delete directory
    #     # shutil.rmtree(tmp_dir)
    #     shutil.rmtree(tmp_dir[0])  
    # except OSError as exc:
    #     # ENOENT - no such file or directory
    #     if exc.errno != errno.ENOENT:  
    #         raise  IOError("No directory to delete")
    
    print ('Time taken for pickling was', time.time() - t_pickle, 'seconds')

    Dict = loadmat(os.path.join(tmp_dir,'rest'))
    MainData.C = int(Dict['C'])
    MainData.ndim = int(Dict['ndim'])
    MainData.nvar = int(Dict['nvar'])

    return MainData, mesh, material, Eulerx, TotalPot





class constant_camera_view(object):
    """Keeps the camera view for mayavi animations fixed
    """

    def __init__(self):

        import os
        os.environ['ETS_TOOLKIT'] = 'qt4'
        from mayavi import mlab

        self.mlab = mlab

    def __enter__(self):
        mlab = self.mlab
        self.orig_no_render = mlab.gcf().scene.disable_render
        if not self.orig_no_render:
            mlab.gcf().scene.disable_render = True
        cc = mlab.gcf().scene.camera
        self.orig_pos = cc.position
        self.orig_fp = cc.focal_point
        self.orig_view_angle = cc.view_angle
        self.orig_view_up = cc.view_up
        self.orig_clipping_range = cc.clipping_range

    def __exit__(self, t, val, trace):
        mlab=self.mlab
        cc = mlab.gcf().scene.camera
        cc.position = self.orig_pos
        cc.focal_point = self.orig_fp
        cc.view_angle =  self.orig_view_angle 
        cc.view_up = self.orig_view_up
        cc.clipping_range = self.orig_clipping_range

        if not self.orig_no_render:
            mlab.gcf().scene.disable_render = False
        if t != None:
            print(t, val, trace)
            ipdb.post_mortem(trace)