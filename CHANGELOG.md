# Change Log
**V0.1rc02**
This release brings outstanding changes to florence.

- Finally a much cleaner higher order mesh generator for tris and tets using pure serial numpy solution. It doesn't get faster than this, as the cost is reduced to dgemm calls and further finding uniques along rows which is O(n x logn). Generating 10.5 million nodes (for p=3) takes 3 minutes with this algorithm. With the older parallel algorithm which was still fast the same mesh took 5.5 days, essentially because it had an O(n^2) cost.
- A much more memory efficient assembler. Many data types have been silently changed to numpy.float32/numpy.int32. Explicit calls to gc.collect() and del. For memory efficiency it is a good practice to turn off parallelisation.
- Out of core assembly of matrices using dask library. Out of core sparse matrix format from repo https://github.com/aidan-plenert-macdonald/scipy/tree/master/scipy/dsparse. 
- Orders of magnitude speed improvement in Mesh class. All mesh class routines are now optimised to their full serial efficiency.
- Extremely fast 3D curvilinear visualisation method under PostProcess. Millions of data points can be plotted in a few seconds. The data points now go through a VTK filter, so now no warnings are generated regarding normals.
- 2D solutions for curvilinear mesh generator for planar faces.
- Arc length based projection in 3D is now implemented and is working, although in the event of unsuccessful projection it falls back to orthogonal projection.
- And above all the tests pass i.e. linear/non-linear analyses, 2D/3D curvilinear meshings, solvers, examples etc.

Note that finally this is also maintenance release as the code requires significant clean-up and many deprecated features would be removed right in the next few commits.    