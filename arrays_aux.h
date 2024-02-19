#include "thread_pool.h"
/*create a global pool*/
const int nb_threads = 30;
ThreadPool pool(nb_threads);
const bool parallel_array = false;

/*return a vector of n random values between 0 and one*/
std::vector<float> random_values(size_t n) {
    std::vector<float> res(n);
    std::default_random_engine generator(std::time(nullptr));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < n; ++i) {
        res[i] = distribution(generator);
    }
    return res;
}

/*define a function whitch applay an other one a specific range and store the ressult given in the input vector*/
template <typename R, typename F>
void apply_range(size_t start, size_t end, std::vector<R> &store, F func) {
    //std::cout << "applying range : " << start << " to " << end << std::endl;
    /*if (start == 0){
        std::cout << "applying range : " << start << " to " << end << std::endl;
    }*/
    for (size_t i = start; i < end; ++i) {
        store[i] = func(i);
        //std::cout << i <<": " <<func(i) <<" " << store[i] << std::endl;
    }
}
/*define a void function whitch applay an other void one a specific range*/
template <typename F>
void apply_range_void(size_t start, size_t end, F func) {
    for (size_t i = start; i < end; ++i) {
        func(i);
    }
}