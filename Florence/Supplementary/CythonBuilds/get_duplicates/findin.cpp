#include <vector>
#include <algorithm>
#include <omp.h>

inline std::vector<long int> FindEqual(const long int *arr, long int size,long int num) {
    std::vector<long int> counts;
    auto p = std::find(arr,arr+size,num);
    if (p-arr==size) {
        return counts;
    }
    for (auto i=0; i< size;i++) {
         auto dist = p-arr+1;
         p = std::find(arr+dist,arr+size,num);
         counts.push_back(dist-1);
         if (p-arr==size)
             break;
    }
    return counts;
}


inline std::vector<std::vector<long int> > get_indices_cpp(long int *inv, long int *unique_inv, long int size_inv, long int size_unique_inv)
{

    std::vector<std::vector<long int> > v;
    std::vector<long int> vec_3, vec_4;

    #pragma omp parallel
    {
        std::vector<long> vec, vec_1, vec_2;
        #pragma omp for nowait //fill vec_private in parallel
        for (auto i=0; i<size_unique_inv; ++i)
        {   
            vec = FindEqual(inv,size_inv,unique_inv[i]);
            if (vec.size() > 1)
            {
                for (size_t j=0; j<vec.size()-1; ++j) {
                    vec_1.push_back(vec[0]);
                }
                for (size_t j=1; j<vec.size(); ++j) {
                    vec_2.push_back(vec[j]);
                }
            }
        }

        #pragma omp critical
        {
            vec_3.insert(vec_3.end(), vec_1.begin(), vec_1.end());
            vec_4.insert(vec_4.end(), vec_2.begin(), vec_2.end());
            v.push_back(vec_3);
            v.push_back(vec_4);
        }
    }
        

    return v;   
}

    