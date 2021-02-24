import numpy as np
import os

import pyircp
from pyzview import Pyzview

if __name__ == '__main__':
    base_folder = os.path.dirname(__file__)
    src = np.fromfile(base_folder + "/test_data/src.bin", dtype=np.float32).reshape((-1, 3))
    dst_all = np.fromfile(base_folder + "/test_data/dst.bin", dtype=np.float32).reshape((-1, 6))
    dst = dst_all[:, :3]
    dst_n = dst_all[:, 3:]

    params = pyircp.ICPParams()

    params.maxIterations = 10
    params.maxNonIncreaseIterations = 3
    params.maxMatchingDistance = 1.0

    params.ransacIterations = 1000
    params.ransacMaxInlierDistance = 0.02
    params.seed = 0

    icp = pyircp.ICP(params)
    res = icp.run(src, dst, dst_n, np.eye(3), np.zeros(3))
    src_p=src@res.rotation.T+res.translation
    Pyzview().add_mesh("src", src.reshape(480,640,3), 'r',alpha=0.9)
    Pyzview().add_mesh("dst", dst.reshape(480,640,3), 'g',alpha=0.9)
    Pyzview().add_mesh("src_p", src_p.reshape(480, 640, 3), 'r', alpha=0.9)
