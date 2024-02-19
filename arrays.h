#include "arrays_aux.h"

/*define array to use it like numpy ones*/
/*array heve an aray, a sahpe and a number of dimention*/
/*it's template is with the type of its unit and it's shape*/
template <typename T,size_t ndim>

class Array {
private:
    std::vector<T> data;
    size_t shape[ndim];
public:
    Array(std::vector<T> data, size_t shape_param[ndim]) : data(data) {
        std::copy(shape_param, shape_param + ndim, shape);
    }
    /*array reation if shape is gven as const*/
    Array(std::vector<T> data, const size_t shape_param[ndim]) : data(data) {
        std::copy(shape_param, shape_param + ndim, shape);
    }
    Array(size_t shape_param[ndim]){
        std::copy(shape_param, shape_param + ndim, shape);
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i) {
            size *= shape[i];
        }
        data = std::vector<T>(size);
    }
    /*array reation if shape is gven as const*/
    Array(const size_t shape_param[ndim]){
        std::copy(shape_param, shape_param + ndim, shape);
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i) {
            size *= shape[i];
        }
        data = std::vector<T>(size);
    }
    /*array from array(copy)*/
    Array(const Array<T, ndim>& other) : data(other.data) {
        std::copy(other.shape, other.shape + ndim, shape);
    }
    /*return a vector of the data*/
    std::vector<T> to_std_vector() const {
        return data;
    }
    /*return the shape*/
    const size_t* get_shape() const {
        return shape;
    }
    /*return the number of dimention*/
    size_t get_ndim() const {
        return ndim;
    }
    /*return the size of the array*/
    size_t size() const {
        return data.size();
    }
    /*define a .set(vector cordinates, val)*/
    void set(size_t index, T val) {
        data[index] = val;
    }
    
    T get(size_t index) {
        return data[index];
    }
    template <typename R>
    Array<T, ndim> operator+(const Array<R, ndim>& other) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] + other.data[i];
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [&other](std::pair<size_t,T> p) {
                return p.second + other.data[p.first];
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim>& operator+=(const Array<R, ndim>& other) {
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] += other.data[i];
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                data[p.first] += other.data[p.first];
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator+(const R scallar) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] + scallar;
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [scallar](std::pair<size_t,T> p) {
                return p.second + scallar;
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim>& operator+=(const R scallar) {
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] += scallar;
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [scallar, this](std::pair<size_t,T> p) {
                data[p.first] += scallar;
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator-(const Array<R, ndim>& other) const {
        if(!parallel_array){
        //normal
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] - other.data[i];
        }
        return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                return p.second - other.data[p.first];
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim> operator-=(const Array<R, ndim>&  other) {
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= other.data[i];
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                data[p.first] -= other.data[p.first];
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator-(const R scallar) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] - scallar;
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [scallar](std::pair<size_t,T> p) {
                return p.second - scallar;
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim>& operator-=(const R scallar) {
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= scallar;
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [scallar,this](std::pair<size_t,T> p) {
                data[p.first] -= scallar;
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator*(const Array<R, ndim>& other) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] * other.data[i];
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                return p.second * other.data[p.first];
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim> operator*=(const Array<R, ndim>& other){
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] *= other.data[i];
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                data[p.first] *= other.data[p.first];
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator*(R scallar) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] * scallar;
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [scallar](std::pair<size_t,T> p) {
                return p.second * scallar;
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim>& operator*=(R scallar) {
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] *= scallar;
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [scallar,this](std::pair<size_t,T> p) {
                data[p.first] *= scallar;
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator/(const Array<R, ndim>& other) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            std::vector<R> other_data= other.to_std_vector();
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] / other_data[i];
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::vector<R> other_data= other.to_std_vector();
            std::function<T(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                return p.second / other_data[p.first];
            };
            return apply_par(func);
        }

    }
    template <typename R>
    Array<T, ndim> operator/=(const Array<R, ndim>& other){
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] /= other.data[i];
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
                data[p.first] /= other.data[p.first];
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator/(R scallar) const {
        if(!parallel_array){
            //normal
            Array<T, ndim> res(shape);
            for (size_t i = 0; i < data.size(); ++i) {
                res.data[i] = data[i] / scallar;
            }
            return res;
        }
        else{
            //parallel
            /*we will call apply_par*/
            std::function<T(std::pair<size_t,T>)> func = [scallar](std::pair<size_t,T> p) {
                return p.second / scallar;
            };
            return apply_par(func);
        }
    }
    template <typename R>
    Array<T, ndim>& operator/=(R scallar) {
        if(!parallel_array){
            //normal
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] /= scallar;
            }
            return *this;
        }
        else{
            //parallel
            /*we will call apply_par_void*/
            std::function<void(std::pair<size_t,T>)> func = [scallar,this](std::pair<size_t,T> p) {
                data[p.first] /= scallar;
            };
            apply_par_void(func);
            return *this;
        }
    }
    template <typename R>
    Array<T, ndim> operator%(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = std::fmod(data[i], other.data[i]);
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator%=(const Array<R, ndim>& other){
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::fmod(data[i], other.data[i]);
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator%(R scallar) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = std::fmod(data[i], scallar);
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator%=(R scallar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::fmod(data[i], scallar);
        }
        return *this;
    }
    Array<T, ndim> operator-() const {
        //normal
        /*
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = -data[i];
        }
        return res;
        */
        //parallel
        /*we will call apply_par*/
        std::function<T(std::pair<size_t,T>)> func = [&](std::pair<size_t,T> p) {
            return -p.second;
        };
        return apply_par(func);
    }
    Array<T, ndim> operator+() const {
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator>(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] > other.data[i]) {
                res.data[i] = 1;
            } else {
                res.data[i] = 0;
            }
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator>=(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] >= other.data[i]) {
                res.data[i] = 1;
            } else {
                res.data[i] = 0;
            }
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator<(const Array<R, ndim>& other) const {
        return other > *this;
    }
    template <typename R>
    Array<T, ndim> operator<=(const Array<R, ndim>& other) const {
        return other >= *this;
    }
    template <typename R>
    Array<T, ndim> operator==(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] == other.data[i]) {
                res.data[i] = 1;
            } else {
                res.data[i] = 0;
            }
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator!=(const Array<R, ndim>& other) const {
        return !(*this == other);
    }
    template <typename R>
    Array<T, ndim> operator&&(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] && other.data[i]) {
                res.data[i] = 1;
            } else {
                res.data[i] = 0;
            }
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator||(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] || other.data[i]) {
                res.data[i] = 1;
            } else {
                res.data[i] = 0;
            }
        }
        return res;
    }
    Array<T, ndim> operator!() const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i]) {
                res.data[i] = 0;
            } else {
                res.data[i] = 1;
            }
        }
        return res;
    }
    /*sum over an axis*/
    Array<T,ndim-1> sum_axis(size_t axis) const{
        size_t new_shape[ndim-1];
        size_t j = 0;
        for (size_t i = 0; i < ndim; ++i) {
            if (i != axis) {
                new_shape[j] = shape[i];
                ++j;
            }
        }
        Array<T,ndim-1> res(new_shape);
        for (size_t i = 0; i < data.size(); ++i) {
            size_t index = i;
            size_t index_res = 0;
            for (size_t j = 0; j < ndim; ++j) {
                if (j != axis) {
                    index_res += (index % shape[j]) * shape[j];
                    index /= shape[j];
                }
            }
            res.data[index_res] += data[i];
        }
        return res;
    }
    /*sum over all axis*/
    T sum() const{
        T res = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            res += data[i];
        }
        return res;
    }
    /*normalized*/
    Array<T,ndim> normalized() const{
        Array<T,ndim> res(shape);
        T sum = this->sum();
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i]/sum;
        }
        return res;
    }
    /*max*/
    T max() const{
        T res = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] > res) {
                res = data[i];
            }
        }
        return res;
    }
    /*min*/
    T min() const{
        T res = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] < res) {
                res = data[i];
            }
        }
        return res;
    }
    /*compule vector of pair(index,frequence) from a list of n random values beetwin 0 and 1*/
    /*firstl sort the randoms values*/
    /*then take the normalized value of a coppy an the array given as input*/
    /*index_arr,sum,index_rand,freq =0
    for index_rand:
        while sum<index_rand:
            if freq>0:
                push pair(index_arr,freq) in the vector
            sum+=index_arr[index]
            index+=1
        freq+=1
    if freq>0:
        push pair(index_arr,freq) in the vector
    return vector*/
    std::vector<std::pair<size_t, size_t>> list_coord_freq(size_t n) const {
        /*std::cout << n << std::endl;*/
        std::vector<float> choice_indices = random_values(n);
        /*sort the randoms values*/
        std::sort (choice_indices.begin(), choice_indices.end());
        std::vector<std::pair<size_t, size_t>> res;
        std::vector<float> p_flat = this->normalized().to_std_vector();
        std::sort(p_flat.begin(), p_flat.end());
        std::vector<float> p_flat_copy = p_flat;
        float sum = p_flat[0];
        size_t index_arr = 0;
        size_t index_rand = 0;
        size_t freq = 0;
        /*std::cout << "randoms:" << std::endl;*/
        /*for (size_t i = 0; i < n; ++i) {
            std::cout << choice_indices[i] << std::endl;
        }*/
        while (index_rand < n) {
            /*std::cout << "index_rand:" << index_rand << std::endl;*/
            while (sum < choice_indices[index_rand]) {
                /*std::cout << "index_arr:" << index_arr << std::endl;
                std::cout << "sum:" << sum << std::endl;*/
                if (freq > 0) {
                    /*std::cout << "index_arr:" << index_arr << " freq:" << freq << std::endl;*/
                    res.push_back(std::make_pair(index_arr, freq));
                    freq = 0;
                }
                ++index_arr;
                sum += p_flat[index_arr];
            }
            ++freq;
            ++index_rand;
        }
        if (freq > 0) {
            /*std::cout << "index_arr:" << index_arr << " freq:" << freq << std::endl;*/
            if (index_arr < p_flat.size()){
                res.push_back(std::make_pair(index_arr, freq));
            }
        }
        return res;
    }
    /*take an int random value for each element folowing the rul normal(n*proba, var) */
    std::vector<std::pair<size_t, size_t>> list_coord_freq2(size_t n) const {
        std::vector<std::pair<size_t, size_t>> res;
        std::vector<float> p_flat = this->normalized().to_std_vector();
        std::vector<float> p_flat_copy = p_flat;
        std::default_random_engine generator(std::time(nullptr));
        for(size_t i = 0; i < p_flat.size(); ++i){
            /*central limit theorem*/
            float sigma = std::sqrt(p_flat[i] * (1 - p_flat[i]));
            float mu = p_flat[i];
            /*std::cout << "sigma:" << sigma << " mu:" << mu << std::endl;
            std::cout << "new_sigma:" << sigma/std::sqrt(n) << " new_mu:" << n*mu << std::endl;*/
            if(mu>0){
                /*std::cout << "distri creation"<< std::endl;*/
                std::normal_distribution<float> distribution(n*mu, sigma/std::sqrt(n));
                /*std::cout << "distri created"<< std::endl;*/
                float val_f = distribution(generator);
                int val = (size_t) val_f;
                /*std::cout << "val:" << val << std::endl;*/
                if (val > 0) {
                    res.push_back(std::make_pair(i, val));
                }
            }
        }
        /*len of res*/
        std::cout << "len of res:" << res.size() << std::endl;
        return res;

    }
    /*apply function f of type T->R to the array*/
    template <typename R>
    Array<R, ndim> apply(R (*f)(T)) const {
        Array<R, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.set(i,f(data[i]));
        }
        return res;
    }
    /*filling function*/
    void fill(T value) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = value;
        }
    }
    /*define the copy to enable returnin an array*/
    Array<T, ndim> copy() const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i];
        }
        return res;
    }

    /*define an operator to apply a function to elements in parallel*/
    template <typename R>
    Array<R, ndim> apply_par(std::function<R(std::pair<size_t,T>)> f) const {
        /*split the array in nb_threads parts*/
        std::vector<R> result_vect (data.size());
        std::vector<size_t> ranges(nb_threads + 1);
        size_t size = data.size();
        for (size_t i = 0; i < nb_threads; ++i) {
            ranges[i] = (size_t) (i * size / nb_threads);
        }
        ranges[nb_threads] = size;
        /*use apply_range function*/
        /*vector of threads*/
        std::vector<std::thread> threads;
        /*transform f such that f only take one size_t parameter*/
        auto f2 = [&](size_t i) {
                return f(std::make_pair(i,data[i]));
        };
        //clasic threads
        /*for (size_t i = 0; i < nb_threads; ++i) {
            threads.push_back(
                        std::thread(
                                apply_range<R, decltype(f2)>,
                                ranges[i],
                                ranges[i + 1],
                                std::ref(result_vect),
                                f2
                        )
                    );
        }
        for (size_t i = 0; i < nb_threads; ++i) {
            threads[i].join();
        }*/
        //using pool
        for (size_t i = 0; i < nb_threads; ++i) {
            //std::cout<<"adding task"<<ranges[i]<<" to "<<ranges[i + 1]<<std::endl;
            //create a void task with no parameter
            pool.add_task(  [i,f2,&ranges,&result_vect](){
                                    apply_range<R, decltype(f2)>(
                                        ranges[i],
                                        ranges[i + 1],
                                        std::ref(result_vect),
                                        f2
                                    );
                                }
                            ,i
                    );
        }
        pool.wait_all_tasks();
        return Array<R, ndim>(result_vect, shape);     
    }
    /*define an operator to apply_void a function to elements in parallel*/
    /*same as before but the function does not return value*/
    void apply_par_void(std::function<void(std::pair<size_t,T>)> f) const {
        /*split the array in nb_threads parts*/
        std::vector<size_t> ranges(nb_threads + 1);
        size_t size = data.size();
        for (size_t i = 0; i < nb_threads; ++i) {
            ranges[i] = (size_t) (i * size / nb_threads);
        }
        ranges[nb_threads] = size;
        /*use apply_range function*/
        /*vector of threads*/
        std::vector<std::thread> threads;
        /*transform f such that f only take one size_t parameter*/
        auto f2 = [&](size_t i) {
                f(std::make_pair(i,data[i]));
        };

        for (size_t i = 0; i < nb_threads; ++i) {
            threads.push_back(
                        std::thread(
                                apply_range_void<decltype(f2)>,
                                ranges[i],
                                ranges[i + 1],
                                f2
                        )
                    );
        }
        for (size_t i = 0; i < nb_threads; ++i) {
            threads[i].join();
        }    
    }
    /*overload the << operator*/
    friend std::ostream& operator<<(std::ostream& os, const Array<T, ndim>& array) {
        os << "Array(";
        for (size_t i = 0; i < array.data.size(); ++i) {
            os << array.data[i];
            if (i != array.data.size() - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};