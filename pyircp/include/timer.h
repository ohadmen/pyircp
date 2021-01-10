#pragma once

#include <chrono>
#include <string>

namespace core {

class Timer
{
public:
    Timer() : start_(std::chrono::high_resolution_clock::now())
    {}
            
    double elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_
        ).count() / 1000.;
    }

    std::string printMs(const std::string& message) const {
        return message + ": " + std::to_string(elapsedMs()) + " ms";
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_; 
};


}
