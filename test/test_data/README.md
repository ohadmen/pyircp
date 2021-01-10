# ICP test point cloud
in this director you can find two files:
* src.bin - source point cloud. Points are stored serially float32 point coordinates (x1,y1,z1, x2,y2,z2....). 
* dst.bin - destination point cloud. Points and normals are stored serially float32 (x1,y1,z1,nx1,ny1,nz1,x2,y2,z2,nx1,nx2,nz2...).
In both cases you will find 640x480=307200 points, some are nan. When reshaped, to 480x640x3, you can extract the meshing.
src file size is 640x480x3x4=3686400Bytes (number of points x float per point x bytes per float).

ground truth transformation between src and dst:

[ 0.99935008, -0.02989301,  0.02014532, -0.2       ]
[ 0.03009299,  0.99950006, -0.0096977 ,  0.1       ]
[-0.01984535,  0.01029763,  0.99975003,  0.3       ]
[ 0.        ,  0.        ,  0.        ,  1.        ]


