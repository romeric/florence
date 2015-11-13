#ifndef STD_INC_HPP
#define STD_INC_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdarg>
#include <cmath>
#include <ctime>
#include <tuple>
#include <typeinfo>
#include <chrono>
#include <memory>
#include <algorithm>
#include <utility>
#include <limits>


typedef int64_t Integer;
typedef uint64_t UInteger;
typedef double Real;
typedef bool Boolean;

#define False false
#define True  true


// Control function inlining more aggressively
#if defined(__GNUC__) || defined(__GNUG__)
    #define ALWAYS_INLINE inline __attribute__((always_inline))
    #define NEVER_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define ALWAYS_INLINE __forceinline
    #define NEVER_INLINE __declspec(noinline)
#endif

#define STATIC static

#define INF std::numeric_limits<double>::infinity()

#endif // STD_INC_HPP

