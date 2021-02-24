import os
import unittest
import time

import pyircp
import numpy as np



def read_float_file(path):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape((-1, 3))


class TestPycuicp(unittest.TestCase):

    def test_result(self):
        r = pyircp.ICPResult()

        self.assertEqual(r.nInliers, 0)
        self.assertFalse(r.valid())

        r.nInliers = 1
        r.status = pyircp.ICPResult.OK
        self.assertTrue(r.valid())

        self.assertEqual(r.nInliers, 1)

    def test_icp_pcl(self):
        base_folder = os.path.dirname(__file__)
        src = read_float_file(base_folder + "/test_data/src.bin")
        dst_all = read_float_file(base_folder + "/test_data/dst.bin")
        dst = dst_all[0::2, :]
        dst_normals = dst_all[1::2, :]

        expected_shape = (640 * 480, 3)

        self.assertEqual(src.shape, expected_shape)
        self.assertEqual(dst.shape, expected_shape)
        self.assertEqual(dst_normals.shape, expected_shape)

        params = pyircp.ICPParams()

        params.maxIterations = 10
        params.maxNonIncreaseIterations = 3
        params.maxMatchingDistance = 1.0

        params.ransacIterations = 1000
        params.ransacMaxInlierDistance = 0.02
        params.seed = 0

        icp = pyircp.ICP(params)

        expected = np.array([
            0.99935008, -0.02989301, 0.02014532, -0.2,
            0.03009299, 0.99950006, -0.0096977, 0.1,
            -0.01984535, 0.01029763, 0.99975003, 0.3,
            0., 0., 0., 1.0
        ])
        expected = expected.reshape((4, 4))
        start = time.time()
        res = icp.run(src, dst, dst_normals, np.eye(3), np.zeros(3))
        end = time.time()
        print(
            "#source:{} #dst:{} execution time: {:5.3f}msec".format(np.sum(~np.isnan(src)) // 3,
                                                               np.sum(~np.isnan(dst)) // 3,
                                                               (end - start) * 1e3))
        self.assertGreater(res.nInliers, 640 * 480 * 0.5)
        self.assertEqual(res.nInliers, len(res.inliers))
        np.testing.assert_almost_equal(res.rotation, expected[:3, :3], decimal=3)
        np.testing.assert_almost_equal(res.translation, expected[:3, 3], decimal=2)


if __name__ == '__main__':
    unittest.main()


