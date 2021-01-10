#pragma once

#include "ioprint.h"

#include <string>
#include <type_traits>
#include <exception>

namespace core {

class Exception : public std::exception
{
public:
    Exception() {}

    template<typename... Args>
    Exception(const Args&... args);

    virtual const char* what() const noexcept override;

    void append(const std::string& w);

    Exception& at(const char* file, int line);

private:
    std::string what_;
};


#define LIKELY(x) __builtin_expect(static_cast<bool>(x), 1)
#define UNLIKELY(x) __builtin_expect(static_cast<bool>(x), 0)

#define REQUIRE_EX(cond, ex, msg, ...) \
    do { \
        if (!LIKELY(cond)) { \
            throw ex(__VA_ARGS__).at(__FILE__, __LINE__) << msg; \
        } \
    } while (false)

#define REQUIRE_DEF(cond, msg)     REQUIRE_EX(cond, ::core::Exception, msg)

#define REQUIRE_GET_NAME(_1, _2, _3, NAME, ...)     NAME
#define REQUIRE(...)    REQUIRE_GET_NAME(__VA_ARGS__, REQUIRE_EX, REQUIRE_DEF)(__VA_ARGS__)


// -------------------------------------------------------------------------------------------------

template<typename T, typename F> 
std::enable_if_t<
    std::is_base_of<Exception, std::remove_reference_t<T>>::value,
    T&&
>
operator<<(T&& e, const F& info)
{
    e.append(::core::ioprint(info));
    return std::forward<T>(e);
}

inline const char* Exception::what() const noexcept
{
    return what_.c_str();
}

template<typename... Args>
Exception::Exception(const Args&... args)
{
    if constexpr(sizeof...(args) > 0)
    {
        (*this << ... << args);
    }
}

inline void Exception::append(const std::string& w)
{
    what_ += w;
}

inline Exception& Exception::at(const char* file, int line)
{
    *this << "at: " << file << "(" << line << "): ";
    return *this;
}

}; //namespace
