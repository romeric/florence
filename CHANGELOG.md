# Change Log

**V0.1**

The first official release of Florence is here. A lot of fundamental changes have been done compared to the beta releases and indeed almost 80% of Florence's functionality was developed since the last beta release, so it is impossible to enumerate the changes and new features. Nevertheless, here are the ones that stand one and make Florence what it is 

1. Florence has a complete C++ level interface (and we really mean *complete*... *this time*) for assembling implicit and explicit FEM problems
2. Yes, on that note, extremely efficient explicit FEM solvers for mechanical and electromechanical problem are available which can solve millions of DoFs in seconds on a single core.
3. Modular explicit penalty contact formulation is available for all types of formulations.
4. Strain gradient, couple stress and micropolar based elasticity and electro-elasticity solvers using mixed low and high order FEM.
5. Florence now supports SIMD, shared parallelism, could/network based parallelism and cluster based parallelism for all types of problems.
6. Support for Python 2 and 3 under Linux, macOS and Windows (under Cygwin).
7. A comprehensive Mesh module. Way too many changes/new features in this module to list.
8. Interface to MUMPS and Pardiso direct sparse solvers.
9. Automatic unit testing and code coverage.
10. Incredibly lean and lightweight source package (~1.8MB). Note that, as a result of this clean up, older *tags* may have been broken. 
11. Parallel build configuration for `setup.py`
12. And, we are finally on PyPI. `pip install Florence`

and enjoy!   

**V0.1rc04**

More fundamental changes in this release:

- Support for Python 3 (CPython 3.6) and PyPy (v5.7.1).
- Major bug fix for nonlinear electromechanics. It should be considered stable now.
- PostMesh is no longer shipped with florence and should be installed externally.
- Dispatch numerical integration of `BDB` to optimised BLAS. With this change the whole numerical integration of all variational formulations become `BLAS/SIMD/C` optimised with no python layer.

**V0.1rc03**

This pre-release brings fundamental performance improvements to florence in particular:
- Low level SIMD dispatcher for implementation and numerical integration of complex multi-variable material models based on sub-project Fastor (https://github.com/romeric/Fastor) 
- SIMD optimised implementation for numerical integration of geometric stiffness matrices for displacement and displacement potential formulations.
- Low level SIMD dispatcher for computing kinematic measures.
- Fast scipy solution for computing the Dirichlet boundary conditions
- Normalised, relative and absolute tolerances for Newton Raphson algorithm
- Stable implemenations for Tris, Tets, Quads and Hexes
- Complete curvilinear plotting and animation engine for all element types

and much more!


**V0.1rc02**

This release brings outstanding changes to florence.

- Finally a much cleaner higher order mesh generator for tris and tets using pure serial numpy solution. It doesn't get faster than this, as the cost is reduced to `dgemm` calls and further finding uniques along rows which is O(n x logn). Generating 10.5 million nodes (for p=3) takes 3 minutes with this algorithm. With the older parallel algorithm which was still fast the same mesh took 5.5 days, essentially because it had an O(n^2) cost.
- A much more memory efficient assembler. Many data types have been silently changed to `numpy.float32`/`numpy.int32`. Explicit calls to `gc.collect()` and `del`. For memory efficiency it is a good practice to turn off parallelisation.
- Out of core assembly of matrices using dask library. Out of core sparse matrix format from repository https://github.com/aidan-plenert-macdonald/scipy/tree/master/scipy/dsparse. 
- Orders of magnitude speed improvement in Mesh class. All mesh class routines are now optimised to their full serial efficiency.
- Extremely fast 3D curvilinear visualisation method under PostProcess. Millions of data points can be plotted in a few seconds. The data points now go through a VTK filter, so now no warnings are generated regarding normals.
- 2D solutions for curvilinear mesh generator for planar faces.
- Arc length based projection in 3D is now implemented and is working, although in the event of unsuccessful projection it falls back to orthogonal projection.
- And above all the tests pass i.e. linear/non-linear analyses, 2D/3D curvilinear meshings, solvers, examples etc.

Note that finally this is also maintenance release as the code requires significant clean-up and many deprecated features would be removed right in the next few commits.    
