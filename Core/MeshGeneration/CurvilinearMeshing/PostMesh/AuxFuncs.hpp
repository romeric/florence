#ifndef AUX_FUNCS_HPP
#define AUX_FUNCS_HPP


#ifndef EIGEN_INC_HPP
#include <EIGEN_INC.hpp>
#endif

//! AUXILARY FUNCTIONS FOR POSTMESH
inline std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    //! SPLIT STRINGS
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

inline std::vector<std::string> split(const std::string &s, char delim) {
    //! SPLIT STRINGS
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


template<typename T>
inline void print(std::vector<T> arr)
{
    //! PRINT FUNCTION OVERLOADED FOR STL VECTORS
    //! EASIER TO BIND EVERYTHING WITH PRINT FUNCTION
    //! RATHER THAN BINDING WITH OPERATOR <<

    //typeid(arr[0]).name();
    std::cout << std::endl;
    for (typename std::vector<T>::const_iterator i=arr.begin(); i<arr.end();++i)
    {
        std::cout <<  *i << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
inline void print(const T last)
{
    //! PRINT FUNCTION OVERLOADED FOR GENERIC TYPE T
    std::cout << last << std::endl;
}

template <typename U, typename... T>
inline void print(const U first, const T... rest)
{
    //! PRINT FUNCTION OVERLOADED USING VARIADIC TEMPLATE ARGUMENTS
    std::cout << first << " ";
    print(rest...);
}

template <typename T>
inline void warn(const T last)
{
    //! WARN FUNCTION FOR GENERIC TYPE T
    std::cerr << last << std::endl;
}

template <typename U, typename... T>
inline void warn(const U first, const T... rest)
{
    //! WARN FUNCTION OVERLOADED USING VARIADIC TEMPLATE ARGUMENTS
    std::cerr << first << " ";
    warn(rest...);
}



template<typename T, typename ... Args>
inline std::chrono::duration<double> timer(T (*func)(Args...), Args...args)
{
    //! Generic timer function for measuring elapsed time on a given
    //! function. For simple functions with no overload, pass the
    //! function to timer function directly, followed by arguments
    //!
    //! for example: Given a function
    //!
    //!     int silly_add(double a, double b)
    //!         return reinterpret_cast<int>(a+b);
    //!
    //! To measure the time spent on this function, call the timer as
    //!
    //!     timer(silly_add,a,b);
    //!
    //! For overloaded functions, since a pointer to an overloaded function
    //! can be ambiguous, you need to use static_cast
    //!
    //! for example: Given a function
    //!
    //!     double simple_mul(double a,double b)
    //!         return a*b;
    //!
    //! with overload
    //!
    //!     double simple_mul(double a, double b, double c)
    //!         return a*b*c;
    //!
    //! call the timer function on the first overload as
    //!
    //!     timer(static_cast<double (*)(double,double)>(&simple_mul),a,b);
    //!
    //! and on the second overload as
    //!
    //!     timer(static_cast<double (*)(double,double,double)>(&simple_mul),a,b,c);
    //!
    //! you can also explicitly create a function pointer and pass it as follows:
    //!
    //!     double (*simple_mul_ptr)(double,double) = &simple_mul;
    //!
    //! then
    //!
    //!     timer(simple_mul_ptr,a,b);



    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    func(args...);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    if (elapsed_seconds.count() >= 1.0e-3 && elapsed_seconds.count() < 1.)
    {
        // Alternatively duration_cast can also be used but decimal digits would be lost
        //std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();
        std::cout << "Elapsed time is "<< elapsed_seconds.count()/1.0e-03 << " ms" << std::endl;
    }
    else if (elapsed_seconds.count() >= 1.0e-6 && elapsed_seconds.count() < 1.0e-3)
    {
        // Alternatively duration_cast can also be used but decimal digits would be lost
        //std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count();
        std::cout << "Elapsed time is "<< elapsed_seconds.count()/1.0e-06 << " \xC2\xB5s" << std::endl;
    }
    else if (elapsed_seconds.count() < 1.0e-6)
    {
        // Alternatively duration_cast can also be used but decimal digits would be lost
        //std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_seconds).count();
        std::cout << "Elapsed time is "<< elapsed_seconds.count()/1.0e-09 << " ns" << std::endl;
    }
    else
    {
        // Alternatively duration_cast can also be used but decimal digits would be lost
        //std::chrono::duration_cast<std::chrono::seconds>(elapsed_seconds).count();
        std::cout << "Elapsed time is "<< elapsed_seconds.count()<< " s" << std::endl;
    }

    return elapsed_seconds;
}


template<typename T, typename ... Args>
inline double timeit(T (*func)(Args...), Args...args)
{
    //! IMPORTANT: Do not pass function to timeit which modify their input
    //! arguments (unless you are completely sure, it does not affect the timing),
    //! since the same function with modified arguments will have a different run-time.
    //!
    //! Generic timer function for accurately measuring elapsed time on a
    //! given function. timeit runs a function many times and gives the mean of all
    //! run time. For simple functions with no overload, pass the
    //! function to timer function directly, followed by arguments
    //!
    //! for example: Given a function
    //!
    //!     int silly_add(double a, double b)
    //!         return reinterpret_cast<int>(a+b);
    //!
    //! To measure the time spent on this function, call the timeit as
    //!
    //!     timeit(silly_add,a,b);
    //!
    //! For overloaded functions, since a pointer to an overloaded function
    //! can be ambiguous, you need to use static_cast
    //!
    //! for example: Given a function
    //!
    //!     double simple_mul(double a,double b)
    //!         return a*b;
    //!
    //! with overload
    //!
    //!     double simple_mul(double a, double b, double c)
    //!         return a*b*c;
    //!
    //! call the timeit function on the first overload as
    //!
    //!     timeit(static_cast<double (*)(double,double)>(&simple_mul),a,b);
    //!
    //! and on the second overload as
    //!
    //!     timeit(static_cast<double (*)(double,double,double)>(&simple_mul),a,b,c);
    //!
    //! you can also explicitly create a function pointer and pass it as follows:
    //!
    //!     double (*simple_mul_ptr)(double,double) = &simple_mul;
    //!
    //! then
    //!
    //!     timeit(simple_mul_ptr,a,b);

    double counter = 1.0f;
    double mean_time = 0.0f;
    for (auto iter=0; iter<1e09; ++iter)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();

        func(args...);

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        mean_time += elapsed_seconds.count();
        counter++;

        if (mean_time > 0.1)
        {
            mean_time /= counter;
            if (mean_time >= 1.0e-3 && mean_time < 1.)
            {
                std::cout << static_cast<long int>(counter)<< " runs, average elapsed time is "<< mean_time/1.0e-03 << " ms" << std::endl;
            }
            else if (mean_time >= 1.0e-6 && mean_time < 1.0e-3)
            {
                std::cout << static_cast<long int>(counter)<< " runs, average elapsed time is "<< mean_time/1.0e-06 << " \xC2\xB5s" << std::endl;
            }
            else if (mean_time < 1.0e-6)
            {
                std::cout << static_cast<long int>(counter)<< " runs, average elapsed time is "<< mean_time/1.0e-09 << " ns" << std::endl;
            }
            else
            {
                std::cout << static_cast<long int>(counter)<< " runs, average elapsed time is "<< mean_time << " s" << std::endl;
            }

            break;
        }
    }
    return mean_time;
}



#endif // AUX_FUNCS_HPP

