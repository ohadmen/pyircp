#pragma once

#include <sstream>

namespace core {

template<typename T>
std::string ioprint(const T& v);


// -------------------------------------------------------------------------------------------------

template<>
inline std::string ioprint(const std::string& v)
{
    return v;
}

template<typename T>
std::string ioprint(const T& v)
{
    std::stringstream buf;
    buf.precision(20);
    buf << v;
    return buf.str();
}


}
