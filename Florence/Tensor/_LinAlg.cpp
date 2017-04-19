
#include <_Numeric.cpp>

// REVERSE CUTHILL-MCKEE ALGORITHM
template<typename Integer>
static inline std::vector<Integer> node_degrees(const Integer *ind,
        const Integer *ptr, Integer num_rows) {

    std::vector<Integer> degree(num_rows);

    for (auto ii=0; ii<num_rows; ++ii) {
        degree[ii] = ptr[ii + 1] - ptr[ii];
        for (auto jj=ptr[ii]; jj<ptr[ii + 1]; ++jj) {
            if (ind[jj] == ii) {
                // add one if the diagonal is in row ii
                degree[ii] += 1;
                break;
            }
        }
    }
    return degree;
}


template<typename Integer>
// std::vector<Integer> 
void reverse_cuthill_mckee(const Integer *ind, const Integer *ptr, Integer num_rows, Integer *order) {
    //! Reverse Cuthill-McKee ordering of a sparse csr or csc matrix.

    Integer N, N_old, seed, level_start, level_end;
    Integer i, j, ll, level_len, temp, temp2;
    N = 0;

    // std::vector<Integer> order(num_rows);
    std::vector<Integer> degree = node_degrees(ind, ptr, num_rows);
    std::vector<std::size_t> inds = argsort(degree);
    std::vector<std::size_t> rev_inds = argsort(inds);
    auto max = std::max_element(degree.begin(),degree.end());
    std::vector<Integer> temp_degrees(*max);


    // loop over zz takes into account possible disconnected graph.
    for (auto zz=0; zz<num_rows; ++zz) {
        if (int(inds[zz]) != -1) {  // Do BFS with seed=inds[zz]
            seed = inds[zz];
            order[N] = seed;
            N += 1;
            inds[rev_inds[seed]] = -1;
            level_start = N - 1;
            level_end = N;

            while (level_start < level_end) {
                for (auto ii=level_start; ii<level_end; ++ii) {
                    i = order[ii];
                    N_old = N;

                    // add unvisited neighbors
                    for (auto jj=ptr[i]; jj<ptr[i + 1]; ++jj) {
                        // j is node number connected to i
                        j = ind[jj];
                        if (int(inds[rev_inds[j]]) != -1) {
                            inds[rev_inds[j]] = -1;
                            order[N] = j;
                            N += 1;
                        }
                    }

                    // Add values to temp_degrees array for insertion sort
                    level_len = 0;
                    for (auto kk=N_old; kk<N; ++kk) {
                        temp_degrees[level_len] = degree[order[kk]];
                        level_len += 1;
                    }

                    // Do insertion sort for nodes from lowest to highest degree
                    for (auto kk=1; kk<level_len; ++kk) {
                        temp = temp_degrees[kk];
                        temp2 = order[N_old+kk];
                        ll = kk;
                        while ((ll > 0) && (temp < temp_degrees[ll-1])) {
                            temp_degrees[ll] = temp_degrees[ll-1];
                            order[N_old+ll] = order[N_old+ll-1];
                            ll -= 1;
                        }
                        temp_degrees[ll] = temp;
                        order[N_old+ll] = temp2;
                    }
                }

                // set next level start and end ranges
                level_start = level_end;
                level_end = N;
            }
        }

        if (N == num_rows) {
            break;
        }
    }

    // return reversed order for RCM ordering
    // std::reverse(order.begin(),order.end());
    std::reverse(order,order+num_rows);
    // return order;
}
