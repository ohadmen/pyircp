#include "icp_wrapper.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

namespace {

void bindICPParams(py::module& m)
{
    using C = alg::ICPParams;

    py::class_<C>(m, "ICPParams")
        .def(py::init<>())
        .def_readwrite("maxIterations", &C::maxIterations)
        .def_readwrite("maxNonIncreaseIterations", &C::maxNonIncreaseIterations)
        .def_readwrite("maxMatchingDistance", &C::maxMatchingDistance)
        .def_readwrite("ransacIterations", &C::ransacIterations)
        .def_readwrite("ransacMaxInlierDistance", &C::ransacMaxInlierDistance)
        .def_readwrite("seed", &C::seed);
}


void bindICPResult(py::module& m)
{
    using C = alg::ICPResult;

    py::class_<C> cls(m, "ICPResult");

    cls
        .def(py::init<>())
        .def_readwrite("status", &C::status)
        .def_readwrite("nInliers", &C::nInliers)
        .def_readwrite("rotation", &C::rotation)
        .def_readwrite("translation", &C::translation)
        .def_readwrite("inliers", &C::inliers)
        .def("valid", &C::valid);

    py::enum_<C::Status>(cls, "Status")
        .value("OK", C::Status::OK)
        .value("TOO_FEW_INPUT_POINTS", C::Status::TOO_FEW_INPUT_POINTS)
        .value("TOO_FEW_MATCHED_POINTS", C::Status::TOO_FEW_MATCHED_POINTS)
        .export_values();
}

void bindICP(py::module& m)
{
    using C = ICPWrapper;

    py::class_<C>(m, "ICP")
        .def(py::init<const alg::ICPParams&>())
        .def("run", &C::run);
}

}

PYBIND11_MODULE(pyircp, m)
{
    bindICPParams(m);
    bindICPResult(m);
    bindICP(m);
}

