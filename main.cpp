#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <string>
#include <time.h>




#include "ray_tracer.h"




int main() {
    size_t width =150;
    size_t height = 150;

    RayTracer ray_tracer(width, height);
    /*left- right+/ up- down+/ front+ back-*/
    ray_tracer.add_object(Vector3D(0, -1, -1), 0.9f, Color(1, 0.1, 0.0), Material(0, 0), 1);
    ray_tracer.add_object(Vector3D(-1, -1, -1), 0.5f, Color(0, 1, 0), Material(0, 0), 1);
    ray_tracer.add_object(Vector3D(1, -1, -0.5), 0.3f, Color(0, 0, 1), Material(0, 0), 1);
    ray_tracer.add_object(Vector3D(0, 1000, 0), 998.0f, Color(1, 0.2, 0.2), Material(1, 100), 0);
    ray_tracer.add_object(Vector3D(0, 0, 1), 0.2f, Color(0.9, 0.9, 1), Material(0.9, 0.2), 0);
    ray_tracer.add_object(Vector3D(1, -1, 2), 1.0f, Color(0.9, 0.9, 1), Material(0.9, 0.2), 0);
    /*ray_tracer.add_object(Vector3D(-1, -1, 2), 0.9f, Color(0, 0, 1), Material(1, 10), 0);*/
    /*ray_tracer.add_object(Vector3D(0, -10, 15), 2.0f, Color(1, 1, 1), Material(0, 0), 0.5);*/

    /*
    ray_tracer.add_object(Vector3D(0, 1000, 0), 998.0f, Color(1, 1, 1), Material(1, 100), 0);
    ray_tracer.add_object(Vector3D(0, -10, 8), 1.f, Color(1, 1, 1), Material(0, 0), 1);
    ray_tracer.add_object(Vector3D(0, -1, 8), 0.5f, Color(1, 1, 1), Material(1, 100), 0);
    */

    /*add a big object lighting object in front of the cam*/
    /*ray_tracer.add_object(Vector3D(0, 0, 10), 5.0f, Color(1, 0.2, 0.2), Material(0, 0), 1);/*
    /*add a big mirror object in fron of the cam*/
    /*ray_tracer.add_object(Vector3D(0, 0, 10), 5.0f, Color(0.9, 0.9, 0.9), Material(1, 0), 0);*/ 
    /*add a big light begind the cam*/
    /*ray_tracer.add_object(Vector3D(0, 0, -10), 3.0f, Color(1, 0.5, 0.5), Material(0, 0), 1);*/
    // Call render method and perform other operations
    

    /*create a reference image*/    
    std::vector<double> perf_hist_null;
    std::vector<float> times_hist_null;
    std::vector<float> sample_hist_null;
    size_t shape[] = {height, width};
    auto vcol = std::vector<Color>(height * width);
    for(int i = 0; i < height * width; ++i) {
        vcol[i] = Color(0, 0, 0);
    }
    Array<Color,2> compare_null(vcol, shape);
Array<Color,2> image = ray_tracer.render(perf_hist_null, times_hist_null, sample_hist_null, compare_null, false, false, 10, 2000);//00
    //std::cout << "reference image created" << std::endl;
    /*save the result*/
save_image(ray_tracer.image_normalisation(image), "Monte Carlo Ray Tracing reference.txt");
    //std::cout << "reference image saved" << std::endl;
    /*create an immage with classic method and save metrics*/
    std::vector<double> perf_hist_classic;
    std::vector<float> times_hist_classic;
    std::vector<float> sample_hist_classic;
Array<Color,2> image_classic = ray_tracer.render(perf_hist_classic, times_hist_classic, sample_hist_classic, image, false, true, 1, 100);    
    //std::cout << "classic image created" << std::endl;
    /*save the result*/
save_image(ray_tracer.image_normalisation(image_classic), "Monte Carlo Ray Tracing classic.txt");
    //std::cout << "classic image saved" << std::endl;
    /*save metrics*/
save_metrics(perf_hist_classic, times_hist_classic, sample_hist_classic, "Monte Carlo Ray Tracing classic metrics.txt");
    //std::cout << "classic metrics saved" << std::endl;
    /*create an immage with bandit method and save metrics*/
    std::vector<double> perf_hist_bandit;
    std::vector<float> times_hist_bandit;
    std::vector<float> sample_hist_bandit;
Array<Color,2> image_bandit = ray_tracer.render(perf_hist_bandit, times_hist_bandit, sample_hist_bandit, image, true, true, 25, 4);
    //std::cout << "bandit image created" << std::endl;
    /*save the result*/
save_image(ray_tracer.image_normalisation(image_bandit), "Monte Carlo Ray Tracing bandit.txt");
    //std::cout << "bandit image saved" << std::endl;
    /*save metrics*/
save_metrics(perf_hist_bandit, times_hist_bandit, sample_hist_bandit, "Monte Carlo Ray Tracing bandit metrics.txt");
    //std::cout << "bandit metrics saved" << std::endl;
    pool.clear_tasks();
    return 0;
}
int main2() {
    //test parallel array operations
    size_t shape2[] = {20};
    std::vector<float> a = std::vector<float>(20, 0);
    std::vector<float> b = std::vector<float>(20, 0);
    std::vector<float> c = std::vector<float>(20, 0);
    for (int i = 0; i < 20; ++i) {
        a[i] = i;
        b[i] = i*5-6;
        c[i] =  5 + i%8;
    }
    Array<float,1> aa = Array<float,1>(a, shape2);
    Array<float,1> ab = Array<float,1>(b, shape2);
    Array<float,1> ac = Array<float,1>(c, shape2);
    Array<float,1> ad = aa + ab;// + ac;
    std::vector<float> d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] + b[i] ) {//+ c[i]
            std::cout << "error: parallel array operations +" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << " b[i]: " << b[i] << std::endl;//<< " c[i]: " << c[i]
            std::cout << "i: " << i << std::endl;
        }
    }
    
    ad = aa - ab - ac;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] - b[i] - c[i]) {
            std::cout << "error: parallel array operations -" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << " b[i]: " << b[i] << " c[i]: " << c[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ad = aa * ab * ac;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] * b[i] * c[i]) {
            std::cout << "error: parallel array operations *" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << " b[i]: " << b[i] << " c[i]: " << c[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ad = (aa / ab) / ac;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != (a[i] / b[i]) / c[i]) {
            std::cout << "error: parallel array operations /" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << " b[i]: " << b[i] << " c[i]: " << c[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    
    ad = aa + 5;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] + 5) {
            std::cout << "error: parallel array operations + 5" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ad = aa - 5;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] - 5) {
            std::cout << "error: parallel array operations - 5" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ad = aa * 5;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] * 5) {
            std::cout << "error: parallel array operations * 5" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ad = aa / 5;
    d = ad.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        if (d[i] != a[i] / 5) {
            std::cout << "error: parallel array operations / 5" << std::endl;
            std::cout << "d[i]: " << d[i] << " a[i]: " << a[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }

    std::vector<float> ev = std::vector<float>(20, 0);
    for (int i = 0; i < 20; ++i) {
        ev[i] = i;
    }
    Array<float,1> ae = Array<float,1>(ev, shape2);
    std::vector<float> e = ae.to_std_vector();
    ae+=aa;
    e = ae.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        ev[i] += a[i];
        if (e[i] != ev[i]) {
            std::cout << "error: parallel array operations += 5" << std::endl;
            std::cout << "e[i]: " << e[i] << " ev[i]: " << ev[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ae-=aa;
    e = ae.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        ev[i] -= a[i];
        if (e[i] != ev[i]) {
            std::cout << "error: parallel array operations -= 5" << std::endl;
            std::cout << "e[i]: " << e[i] << " ev[i]: " << ev[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ae*=aa;
    e = ae.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        ev[i] *= a[i];
        if (e[i] != ev[i]) {
            std::cout << "error: parallel array operations *= 5" << std::endl;
            std::cout << "e[i]: " << e[i] << " ev[i]: " << ev[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }
    ae/=aa;
    e = ae.to_std_vector();
    for (int i = 0; i < 20; ++i) {
        ev[i] /= a[i];
        if (e[i] != ev[i]) {
            std::cout << "error: parallel array operations /= 5" << std::endl;
            std::cout << "e[i]: " << e[i] << " ev[i]: " << ev[i] << std::endl;
            std::cout << "i: " << i << std::endl;
        }
    }



    std::cout << "parallel array operations test passed" << std::endl;
    


    std::cout << "color test" << Color(1, 0.5, 0.2) << std::endl;
    //clear the pool
    pool.clear_tasks();

    return 0;
}




