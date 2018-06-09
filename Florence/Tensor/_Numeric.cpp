#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <cstdint>

using Long = std::int64_t;

template<typename T>
std::vector<Long> 
FindEqual(const T *arr, Long size,T num) {
    std::vector<Long> counts;
    auto p = std::find_if(arr,arr+size,[&](T j){return j==num;});
    if (p-arr==size) {
        return counts;
    }
    for (auto i=0; i< size;i++) {
         auto dist = p-arr+1;
         p = std::find_if(arr+dist,arr+size,[&](T j){return j==num;});
         counts.push_back(dist-1);
         if (p-arr==size)
             break;
    }
    return counts;
}

template<typename T>
std::vector<Long> 
FindEqualApprox(const T *arr, Long size,T num, double tolerance) {
    std::vector<Long> counts;
    auto p = std::find_if(arr,arr+size,[&](T j){return std::abs(j - num) < tolerance;});
    if (p-arr==size) {
        return counts;
    }
    for (auto i=0; i< size;i++) {
         auto dist = p-arr+1;
         p = std::find_if(arr+dist,arr+size,[&](T j){return std::abs(j - num)<1.0e-14;});
         counts.push_back(dist-1);
         if (p-arr==size)
             break;
    }
    return counts;
}

template<typename T>
std::vector<Long> 
FindLessThan(const T *arr, Long size,T num) {
    std::vector<Long> counts;
    auto p = std::find_if(arr,arr+size,[&](T j){return j < num;});
    if (p-arr==size) {
        return counts;
    }
    for (auto i=0; i< size;i++) {
         auto dist = p-arr+1;
         p = std::find_if(arr+dist,arr+size,[&](T j){return j < num;});
         counts.push_back(dist-1);
         if (p-arr==size)
             break;
    }
    return counts;
}


template<typename T>
std::vector<Long> 
FindGreaterThan(const T *arr, Long size,T num) {
    std::vector<Long> counts;
    auto p = std::find_if(arr,arr+size,[&](T j){return j > num;});
    if (p-arr==size) {
        return counts;
    }
    for (auto i=0; i< size;i++) {
         auto dist = p-arr+1;
         p = std::find_if(arr+dist,arr+size,[&](T j){return j > num;});
         counts.push_back(dist-1);
         if (p-arr==size)
             break;
    }
    return counts;
}


template <typename T>
inline std::vector<std::size_t> 
argsort(const std::vector<T> &v) {

  // INITIALISE INDICES
  std::vector<std::size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // SORT INDICES BY COMPARING VALUES
  std::sort(idx.begin(), idx.end(),
       [&v](std::size_t i1, std::size_t i2) {return v[i1] < v[i2];});

  return idx;
}


template<typename T>
inline std::tuple<std::vector<T>,std::vector<std::size_t> > 
unique(const std::vector<T> &v, bool return_index=false) {

    if (return_index == false) {
        std::vector<T> uniques(v.begin(),v.end());
        std::sort(uniques.begin(),uniques.end());
        uniques.erase(std::unique(uniques.begin(),uniques.end()),uniques.end());

        return std::make_tuple(uniques,std::vector<std::size_t>(0));
    }

    auto sorter = argsort(v);
    auto last = std::unique(sorter.begin(),sorter.end(),[&v](T a, T b){return v[a]==v[b];});
    sorter.erase(last,sorter.end());

    std::vector<T> uniques(sorter.size());
    auto counter = 0;
    for (auto &k: sorter) {
        uniques[counter] = v[k];
        counter++;
    }

    return std::make_tuple(uniques,sorter);
}