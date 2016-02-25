#include <algorithm>
#include <vector>

template<typename T>
std::vector<long int> FindEqual(const T *arr, long int size,T num) {
    std::vector<long int> counts;
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
std::vector<long int> FindEqualApprox(const T *arr, long int size,T num, double tolerance) {
    std::vector<long int> counts;
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
std::vector<long int> FindLessThan(const T *arr, long int size,T num) {
    std::vector<long int> counts;
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
std::vector<long int> FindGreaterThan(const T *arr, long int size,T num) {
    std::vector<long int> counts;
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
std::vector<std::size_t> argsort(const std::vector<T> &v) {

  // INITIALISE INDICES
  std::vector<std::size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // SORT INDICES BY COMPARING VALUES
  std::sort(idx.begin(), idx.end(),
       [&v](std::size_t i1, std::size_t i2) {return v[i1] < v[i2];});

  return idx;
}
