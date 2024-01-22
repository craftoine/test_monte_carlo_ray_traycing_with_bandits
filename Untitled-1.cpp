/*from time import sleep, time
import math
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
class Color(np.ndarray):
    def __new__(cls, r, g, b):
        obj = np.array([r, g, b]).view(cls)
        return obj
class Material:
    def __init__(self, reflection_coefficient, diffuse_coefficient):
        self.reflection_coefficient = reflection_coefficient
        self.diffuse_coefficient = diffuse_coefficient

class SceneObject:
    def __init__(self, position, radius, color, material, emission):
        self.position = position
        self.radius = radius
        self.color = color
        self.material = material
        self.emission = emission

def image_normalisation(image):
    logtransformfactor = 50
    if logtransformfactor != 0:
        img = image * logtransformfactor
        img = np.log(img + 1)
    img = img / np.max(img)
    return img

class RayTracer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scene_objects = []
        self.max_bounces = 7

    def add_object(self, position, radius, color, material, emission):
        self.scene_objects.append(SceneObject(position, radius, color, material, emission))

    def calculate_intersection(self, ray_origin, ray_direction, obj):
        oc = ray_origin - obj.position
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - obj.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            root1 = (-b - math.sqrt(discriminant)) / (2 * a)
            root2 = (-b + math.sqrt(discriminant)) / (2 * a)
            if root1 > 0:
                return ray_origin + ray_direction * root1  # Return hit point, not root1
            elif root2 > 0:
                return ray_origin + ray_direction * root2  # Return hit point, not root2

        return None  # Return None when no intersection is found

    def calculate_surface_normal(self, obj, hit_point):
        return (hit_point - obj.position) / np.linalg.norm(hit_point - obj.position)

    def reflect(self, incident, normal):
        return incident - normal * (2 * np.dot(incident, normal))

    def random_rotate(self, direction, angle_std):
        """phi = np.random.normal(0, angle_std)
        theta = np.random.normal(0, angle_std)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        new_direction = np.array([
            direction[0] * cos_phi + sin_phi * (direction[1] * cos_theta + direction[2] * sin_theta),
            direction[1] * cos_phi - sin_phi * (direction[0] * cos_theta - direction[2] * sin_theta),
            direction[2] * cos_theta - direction[1] * sin_theta
        ])
        return new_direction"""
        initial_angle_theta = atan(direction[1]/direction[0])
        initial_angle_phi = acos(direction.[2])
        angle_phi = initial_angle_phi+distribution(generator)/2;
        angle_theta = initial_angle_theta+distribution(generator);
        cos_phi = cos(angle_phi)
        sin_phi = sin(angle_phi)
        cos_theta = cos(angle_theta)
        sin_theta = sin(angle_theta)
        new_direction = np.array([
            sin_phi * cos_theta,
            sin_phi * sin_theta,
            cos_phi
        ])
        return(new_direction)


    def trace_ray(self, ray_origin, ray_direction):
        ray_color = np.zeros(3)
        energy = np.ones(3)
        for loop in range(self.max_bounces):
            closest_hit = None
            for obj in self.scene_objects:
                hit_point = self.calculate_intersection(ray_origin, ray_direction, obj)
                if hit_point is not None:
                    distance_to_hit = np.linalg.norm(hit_point - ray_origin)
                    if closest_hit is None or distance_to_hit < closest_hit[0]:
                        closest_hit = (distance_to_hit, obj, hit_point)

            if closest_hit is None:
                break
            distance_to_hit, hit_obj, hit_point = closest_hit
            material = hit_obj.material
            normal = self.calculate_surface_normal(hit_obj, hit_point)

            lighting_color = np.zeros(3)
            if hit_obj.emission > 0:
                light_intensity = hit_obj.color * hit_obj.emission
                lighting_color += light_intensity

            reflection_ray_origin = hit_point + normal * 0.001
            reflection_ray_direction = self.reflect(ray_direction, normal)

            if hit_obj.material.diffuse_coefficient != 0:
                new_reflection_ray_direction = self.random_rotate(reflection_ray_direction,
                                                                  hit_obj.material.diffuse_coefficient)
                while np.dot(new_reflection_ray_direction, normal) <= 0:
                    new_reflection_ray_direction = self.random_rotate(reflection_ray_direction,
                                                                      hit_obj.material.diffuse_coefficient)
                reflection_ray_direction = new_reflection_ray_direction

            ray_color += energy * lighting_color
            energy *= hit_obj.color
            energy *= hit_obj.material.reflection_coefficient
            ray_origin = reflection_ray_origin
            ray_direction = reflection_ray_direction
        return ray_color

    def render(self, bandit=False, probas=None, compare_bool=False, compare=None,
               times_hist=None, perf_hist=None, sample_hist=None):
        start_time = time()
        image = np.zeros((self.height, self.width, 3))
        fig, ax = plt.subplots()
        if bandit:
            fig2, ax2 = plt.subplots()
        times = np.zeros((self.height, self.width, 3))
        losts = np.zeros((self.height, self.width))

        for smp in range(samples_per_pixel):
            #nu = np.sqrt(np.log(width*height)/(width*height*samples_per_pixel*subsampling))
            #print("theoritical nu",np.sqrt(np.log(width*height)/(width*height*samples_per_pixel*subsampling)))
            #nu *= 5000#np.sqrt(np.log(width*height)/(width*height*samples_per_pixel*subsampling))
            nu = np.sqrt(np.log(width*height)/(samples_per_pixel))
            #print("second theoritical nu",np.sqrt(np.log(width*height)/(samples_per_pixel)))
            nu *= 400
            print(100*(smp)/samples_per_pixel,"%")
            if smp!=0:
                tm = time()-start_time
                print("start since : ", int(tm)//60,"min",int(tm)%60,"s")
                print("estimated rendering time", int(samples_per_pixel*(tm)/(smp))//60,"min",int(samples_per_pixel*(tm)/(smp))%60,"s")
                print("estimated time left", int((samples_per_pixel-smp)*(tm)/(smp))//60,"min",int((samples_per_pixel-smp)*(tm)/(smp))%60,"s")
            jbs = -1#10
            if not (bandit) or smp ==0:
                newrays = Parallel(n_jobs=jbs)(delayed(self.render_pixel)(x, y,subsampling) for y in range(self.height) for x in range(self.width))
                image += np.array([newrays[i] for i in range(len(newrays))]).reshape((self.height, self.width, 3))               
                times +=np.ones((self.height, self.width,3))*subsampling
            else:
                p_flat = probas.ravel()
                p_flat/=np.sum(p_flat)
                #print(p.max(p_flat)/,np.min(p_flat),np.min(p_flat),np.max(p_flat))
                ind = np.arange(len(p_flat))
                
                res = np.column_stack(
                    np.unravel_index(
                        np.random.choice(ind, p=p_flat, size=subsampling*self.width*self.height),
                        probas.shape))
                #print(res,np.shape(res))
                pixels,nbs = np.unique(res, return_counts=True,axis=0)
                pxs = list(zip(pixels,nbs))
                #print(pxs)
                newrays = Parallel(n_jobs=jbs)(
                                                delayed(self.render_pixel)
                                                (pxs[i][0][1], pxs[i][0][0],pxs[i][1]) for i in range(len(pxs))
                                               )
                m = np.zeros((self.height,self.width,3))
                tms = np.zeros((self.height,self.width,3))
                for i in range(len(pixels)):
                    m[pixels[i][0],pixels[i][1]] = newrays[i]
                    tms[pixels[i][0],pixels[i][1]] = [nbs[i]]*3
                losts +=subsampling*(
                                    np.sum(
                                        (tms>0)*(
                                            1-(np.abs(
                                                image_normalisation(
                                                    (m+image)/(times + tms+0.0001)
                                                )-image_normalisation(
                                                    image/(times+0.0001)
                                                )
                                            )/(tms+0.000001))
                                         )/3
                                        ,axis = 2
                                    )

                                )

                image += m
                times +=tms
                losts-=np.max(losts)#factorise all the value after exp by e^-nu*maxlost so less pressision errors
                print("losts:",np.min(losts),"to",np.max(losts),"shape: ",np.shape(losts))
                exps = np.exp(-nu*losts)
                sm = np.sum(exps)
                probas = exps/sm
                print(np.shape(image))

            if compare_bool:
                times_hist.append(time()-start_time)
                perf_hist.append(np.sum(np.abs(image_normalisation(compare)-image_normalisation(image /(times +0.0001)))**2))
                sample_hist.append((smp+1)*subsampling)
            ax.imshow(image_normalisation(image /(times +0.0001)), origin='upper')
            plt.title(
                "Monte Carlo Ray Tracing results"+str(100*(smp)/samples_per_pixel)+"%"
                )
            plt.xlabel("X")
            plt.ylabel("Y")
            
            if bandit and smp !=0:
                ax2.imshow(probas, origin='upper')
                plt.title(
                "Monte Carlo Ray Tracing probas"+str(100*(smp)/samples_per_pixel)+"%"
                )
            plt.pause(0.01)
            if bandit:
                ax2.clear()
            ax.clear()

        plt.close(fig)
        return image/times

    def render_pixel(self, x, y, subsampling):
        ray_color = np.zeros(3)
        for sub_samp in range(subsampling):
            ray_direction = np.array([((x + random.random() - 0.5) / self.width) - 0.5,
                                      ((y + random.random() - 0.5) / self.width) - 0.5, 1])
            ray_color += self.trace_ray(np.array([0, 0, 0]), ray_direction / np.linalg.norm(ray_direction))
        return ray_color

width=30
height=30
# Setup the scene and render
ray_tracer = RayTracer(width=width, height=height)
ray_tracer.add_object(position=np.array([0, -1, -1]), radius=0.9, color=Color(1, 0.5, 0), material=Material(0, 0), emission=1)
ray_tracer.add_object(position=np.array([-1, -1, -1]), radius=0.5, color=Color(0, 1, 0), material=Material(0, 0), emission=1)
ray_tracer.add_object(position=np.array([0, 1000, 0]), radius=998, color=Color(1, 0.2, 0.2), material=Material(1, 100), emission=0)
ray_tracer.add_object(position=np.array([0, 0, 1]), radius=0.2, color=Color(0.9, 0.9, 1), material=Material(0.9, 0.2), emission=0)
ray_tracer.add_object(position=np.array([1, -1, 2]), radius=1, color=Color(0.9, 0.9, 1), material=Material(0.9, 0.2), emission=0)

logtransformfactor = 50





samples_per_pixel = 5000000
subsampling = 10000
samples_per_pixel//=subsampling


"""image_compare = ray_tracer.render(bandit = False)
np.save("numpy_ref_img_ray", image_compare)"""

image_compare =np.load("numpy_ref_img_ray.npy")
samples_per_pixel = 1000000
subsampling = 25
samples_per_pixel//=subsampling
times_hist_bdt = []
perf_hist_bdt = []
sample_hist_bdt = []
probas = np.ones((height,width))/(width*height)
image_bdt = ray_tracer.render(
                                    bandit = True,
                                    probas = probas,
                                    compare_bool = True,
                                    compare = image_compare,
                                    times_hist = times_hist_bdt,
                                    perf_hist = perf_hist_bdt,
                                    sample_hist = sample_hist_bdt
                                 )
samples_per_pixel = 1000000
subsampling = 2000
samples_per_pixel//=subsampling
times_hist_cla = []
perf_hist_cla = []
sample_hist_cla = []
image_cla = ray_tracer.render(
                                    bandit = False,
                                    compare_bool = True,
                                    compare = image_compare,
                                    times_hist = times_hist_cla,
                                    perf_hist = perf_hist_cla,
                                    sample_hist = sample_hist_cla
                                 )
print(times_hist_cla,perf_hist_cla,sample_hist_cla)
print(times_hist_bdt,perf_hist_bdt,sample_hist_bdt)
def display_image(image):

    plt.imshow(image_normalisation(image), origin='upper')
    plt.title("Monte Carlo Ray Tracing")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

display_image(image_compare)
display_image(image_bdt)
display_image(image_cla)

plt.plot(times_hist_bdt,sample_hist_bdt,label = "bdt")
plt.plot(times_hist_cla,sample_hist_cla,label = "cla")
plt.title("sampling time")
plt.legend()
plt.xlabel("time")
plt.ylabel("sampling")
plt.show()
plt.plot(sample_hist_bdt,perf_hist_bdt,label= "bdt")
plt.plot(sample_hist_cla,perf_hist_cla,label = "cla")
plt.title("differences")
plt.legend()
plt.xlabel("sampling")
plt.ylabel("perf")
plt.show()
*/


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <string>
/*#include "opencv2/opencv.hpp"*/

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

/*define array to use it like numpy ones*/
/*array heve an aray, a sahpe and a number of dimention*/
/*it's template is with the type of its unit and it's shape*/

struct Color {
    float r, g, b;
    Color(float red = 0.0f, float green = 0.0f, float blue = 0.0f) : r(red), g(green), b(blue) {}
    
    Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar);
    }
    Color operator*=(float scalar) {
        r *= scalar;
        g *= scalar;
        b *= scalar;
        return *this;
    }
    Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b);
    }
    Color& operator+=(const Color& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }
    Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b);
    }
    Color& operator*=(const Color& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }
    /*define color>color and < for array definition to work*/
    bool operator>(const Color& other) const {
        return r > other.r && g > other.g && b > other.b;
    }
    bool operator<(const Color& other) const {
        return r < other.r && g < other.g && b < other.b;
    }
    /*define color/color for array definition to work*/
    Color operator/(const Color& other) const {
        return Color(r / other.r, g / other.g, b / other.b);
    }
    /*define color/scalar for array definition to work*/
    Color operator/(float scalar) const {
        return Color(r / scalar, g / scalar, b / scalar);
    }
    /*define color-color for array definition to work*/
    Color operator-(const Color& other) const {
        return Color(r - other.r, g - other.g, b - other.b);
    }
    /*define color-scalar for array definition to work*/
    Color operator-(float scalar) const {
        return Color(r - scalar, g - scalar, b - scalar);
    }

};

struct Vector3D {
    float x, y, z;

    Vector3D(float x_val = 0.0f, float y_val = 0.0f, float z_val = 0.0f) : x(x_val), y(y_val), z(z_val) {}

    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    Vector3D& operator+=(const Vector3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }

    Vector3D operator*(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    Vector3D& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    Vector3D operator*(const Vector3D& other) const {
        return Vector3D(x * other.x, y * other.y, z * other.z);
    }
    Vector3D& operator*=(const Vector3D& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }
    float dot(const Vector3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3D cross(const Vector3D& other) const {
        return Vector3D(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    float magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vector3D normalize() const {
        float mag = magnitude();
        return Vector3D(x / mag, y / mag, z / mag);
    }
    
};
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
    
    void get(size_t index) {
        return data[index];
    }
    template <typename R>
    Array<T, ndim> operator+(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] + other.data[i];
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator+=(const Array<R, ndim>& other) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator+(const R scallar) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] + scallar;
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator+=(const R scallar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += scallar;
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator-(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] - other.data[i];
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator-=(const Array<R, ndim>& other) const {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator-(const R scallar) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] - scallar;
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator-=(const R scallar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= scallar;
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator*(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] * other.data[i];
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator*=(const Array<R, ndim>& other) const {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= other.data[i];
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator*(R scalar) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] * scalar;
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator*=(R scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= scalar;
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator/(const Array<R, ndim>& other) const {
        Array<T, ndim> res(shape);
        std::vector<R> other_data= other.to_std_vector();
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] / other_data[i];
        }
        return res;
    }
    template <typename R>
    Array<T, ndim> operator/=(const Array<R, ndim>& other) const {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= other.data[i];
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator/(R scalar) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] / scalar;
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator/=(R scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= scalar;
        }
        return *this;
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
    Array<T, ndim> operator%=(const Array<R, ndim>& other) const {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::fmod(data[i], other.data[i]);
        }
        return *this;
    }
    template <typename R>
    Array<T, ndim> operator%(R scalar) const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = std::fmod(data[i], scalar);
        }
        return res;
    }
    template <typename R>
    Array<T, ndim>& operator%=(R scalar) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = std::fmod(data[i], scalar);
        }
        return *this;
    }
    Array<T, ndim> operator-() const {
        Array<T, ndim> res(shape);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = -data[i];
        }
        return res;
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

};

struct Material {
    float reflection_coefficient;
    float diffuse_coefficient;
    Material(float reflection = 0.0f, float diffuse = 0.0f) : reflection_coefficient(reflection), diffuse_coefficient(diffuse) {}
};

struct SceneObject {
    Vector3D position;
    float radius;
    Color color;
    Material material;
    float emission;
    SceneObject(Vector3D pos, float rad, Color col, Material mat, float emit) : position(pos), radius(rad), color(col), material(mat), emission(emit) {}
};


/*take an 2d array of color and return an different 2d array of color*/

/*2d array of color to opencv type*/
/*for self as Array<color, 2>*/
/*cv::Mat to_cv_mat(Array<Color, 2> a){
    auto data = a.to_std_vector();
    auto shape = a.get_shape();
    cv::Mat res(shape[0], shape[1], CV_32FC3);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            res.at<cv::Vec3f>(i, j) = cv::Vec3f(data[i * shape[1] + j].r, data[i * shape[1] + j].g, data[i * shape[1] + j].b);
        }
    }
    return res;
}
cv::Mat to_cv_mat(Array<float, 2> a){
    Array<float, 2> b= a.copy();
    b.normalized();
    auto data = b.to_std_vector();
    auto shape = b.get_shape();
    cv::Mat res(shape[0], shape[1], CV_32FC3);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            res.at<cv::Vec3f>(i, j) = cv::Vec3f(data[i * shape[1] + j], data[i * shape[1] + j], data[i * shape[1] + j]);
        }
    }
    return res;
}*/
/*array of color to 0-255 array*/
/*for self as Array<color, 2>*/
Array<unsigned char, 3> to_255(Array<Color, 2> a){
    std::cout << "to_255 from color" << std::endl;
    auto data = a.to_std_vector();
    auto shape = a.get_shape();
    /*vector shape[0],shape[1],3*/
    size_t new_shape[] = {shape[0],shape[1],3};
    std::cout << "shape:" << shape[0] << " " << shape[1] << std::endl;
    auto new_data = std::vector<unsigned char>(shape[0] * shape[1] * 3);
    Array<unsigned char, 3> res(new_data, new_shape);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            /*std::cout << "i:" << i << " j:" << j << std::endl;
            std::cout << "data:" << data[i * shape[1] + j].r << " " << data[i * shape[1] + j].g << " " << data[i * shape[1] + j].b << std::endl ;*/
            res.set(i * shape[1] * 3 + j * 3, (unsigned char)(data[i * shape[1] + j].r * 255));
            res.set(i * shape[1] * 3 + j * 3 + 1, (unsigned char)(data[i * shape[1] + j].g * 255));
            res.set(i * shape[1] * 3 + j * 3 + 2, (unsigned char)(data[i * shape[1] + j].b * 255));
            /*std::cout << "res:" << (unsigned int) res.to_std_vector()[i * shape[1] * 3 + j * 3] << " " << (unsigned int) res.to_std_vector()[i * shape[1] * 3 + j * 3 + 1] << " " << (unsigned int) res.to_std_vector()[i * shape[1] * 3 + j * 3 + 2] << std::endl;*/
        }
    }
    return res;
}
/*for self as Array<float, 2>*/
Array<unsigned char, 3> to_255(Array<float, 2> a){
    auto data = a.to_std_vector();
    auto shape = a.get_shape();
    /*vector shape[0],shape[1],3*/
    size_t new_shape[] = {shape[0],shape[1],3};
    auto new_data = std::vector<unsigned char>(shape[0] * shape[1] * 3);
    Array<unsigned char, 3> res(new_data, new_shape);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            res.set(i * shape[1] * 3 + j * 3, (unsigned char)(data[i * shape[1] + j] * 255));
            res.set(i * shape[1] * 3 + j * 3 + 1, (unsigned char)(data[i * shape[1] + j] * 255));
            res.set(i * shape[1] * 3 + j * 3 + 2, (unsigned char)(data[i * shape[1] + j] * 255));
        }
    }
    return res;
}

/*save a from Array<float, 3> and filename */
void save_image(Array<unsigned char, 3> a, std::string filename){
    auto data = a.to_std_vector();
    auto shape = a.get_shape();
    /*write in a txt file as first line ndim
    second line shape separate by shape
    then shape[0] lines of shape[1] * shape[2] values*/
    /*delete the file*/
    std::remove(filename.c_str());
    /*create the file*/
    /*std::cout << "file opening" << std::endl;*/
    std::ofstream Filetosave(filename);
    if (!Filetosave) {
        std::cout << "Error opening file" << std::endl;
        return;
    }
    /*print file created*/
    /*std::cout << "file created" << std::endl;*/
    /*write the ndim*/
    Filetosave << a.get_ndim() << "\n";
    /*write the shape*/
    for (size_t i = 0; i < a.get_ndim(); ++i) {
        Filetosave << a.get_shape()[i] << " ";
    }
    Filetosave << "\n";
    /*write the data in shape[0] lines*/
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1] * shape[2]; ++j) {
            Filetosave << (unsigned int)data[i * shape[1] * shape[2] + j] << " ";
            /*std::cout << (unsigned int)data[i * shape[1] * shape[2] + j] << " ";*/
        }
        Filetosave << "\n";
        /*std::cout << "\n";*/
    }
    Filetosave.close();
}
void save_image(Array<Color, 2> a, std::string filename){
    auto data = a.to_std_vector();
    auto shape = a.get_shape();
    /*write in a txt file as first line ndim
    second line shape separate by shape
    then shape[0] lines of shape[1] * 3 float values*/
    /*delete the file*/
    std::remove(filename.c_str());
    /*create the file*/
    /*std::cout << "file opening" << std::endl;*/
    std::ofstream Filetosave(filename);
    if (!Filetosave) {
        std::cout << "Error opening file" << std::endl;
        return;
    }
    /*print file created*/
    /*std::cout << "file created" << std::endl;*/
    /*write the ndim*/
    Filetosave << a.get_ndim() << "\n";
    /*write the shape*/
    for (size_t i = 0; i < a.get_ndim(); ++i) {
        Filetosave << a.get_shape()[i] << " ";
    }
    Filetosave << "\n";
    /*write the data in shape[0] lines*/
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            Filetosave << data[i * shape[1] + j].r << " " << data[i * shape[1] + j].g << " " << data[i * shape[1] + j].b << " ";
            /*std::cout << data[i * shape[1] + j].r << " " << data[i * shape[1] + j].g << " " << data[i * shape[1] + j].b << " ";*/
        }
        Filetosave << "\n";
        /*std::cout << "\n";*/
    }
    Filetosave.close();
}
void save_metrics(std::vector<double> perf_hist,std::vector<float> times_hist,std::vector<float> sample_hist,std::string filename){
    /*delete the file*/
    std::remove(filename.c_str());
    /*create the file*/
    /*std::cout << "file opening" << std::endl;*/
    std::ofstream Filetosave(filename);
    if (!Filetosave) {
        std::cout << "Error opening file" << std::endl;
        return;
    }
    /*print file created*/
    /*std::cout << "file created" << std::endl;*/
    /*write the length*/
    Filetosave << perf_hist.size() << "\n";
    /*write the data in 1 lines*/
    for (size_t i = 0; i < perf_hist.size(); ++i) {
        Filetosave << perf_hist[i] << " ";
    }
    Filetosave << "\n";
    /*write the length*/
    Filetosave << times_hist.size() << "\n";
    /*write the data in 1 lines*/
    for (size_t i = 0; i < times_hist.size(); ++i) {
        Filetosave << times_hist[i] << " ";
    }
    Filetosave << "\n";
    /*write the length*/
    Filetosave << sample_hist.size() << "\n";
    /*write the data in 1 lines*/
    for (size_t i = 0; i < sample_hist.size(); ++i) {
        Filetosave << sample_hist[i] << " ";
    }
    Filetosave << "\n";
    Filetosave.close();
}
/*load float as Array<Color,2> from filename*/
Array<Color,2> load_image(std::string filename){
    /*read the file*/
    std::ifstream Filetoread(filename);
    if (!Filetoread) {
        std::cout << "Error opening file" << std::endl;
        /*raise an error*/
        throw std::runtime_error("Error opening file");
    }
    /*read the ndim*/
    size_t ndim;
    Filetoread >> ndim;
    /*read the shape*/
    size_t* shape = new size_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        Filetoread >> shape[i];
    }
    /*read the data*/
    auto data = std::vector<Color>(shape[0] * shape[1]);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float r, g, b;
            Filetoread >> r >> g >> b;
            data[i * shape[1] + j] = Color(r, g, b);
        }
    }
    Filetoread.close();
    return Array<Color,2>(data, shape);
}
/*return max of an image by returning max(max(r), max(g) max (b))*/
float max_image(Array<Color, 2> image){
    float res = 0;
    auto data = image.to_std_vector();
    auto shape = image.get_shape();
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            if (data[i * shape[1] + j].r > res) {
                res = data[i * shape[1] + j].r;
            }
            if (data[i * shape[1] + j].g > res) {
                res = data[i * shape[1] + j].g;
            }
            if (data[i * shape[1] + j].b > res) {
                res = data[i * shape[1] + j].b;
            }
        }
    }
    return res;
}
double mean(Array<Color, 2> image){
    double res = 0;
    auto data = image.to_std_vector();
    auto shape = image.get_shape();
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            res += data[i * shape[1] + j].r;
            res += data[i * shape[1] + j].g;
            res += data[i * shape[1] + j].b;
        }
    }
    return res/(shape[0]*shape[1]*3);
}
double mean(Array<double,2> immage){
    double res = 0;
    auto data = immage.to_std_vector();
    auto shape = immage.get_shape();
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            res += data[i * shape[1] + j];
        }
    }
    return res/(shape[0]*shape[1]);
}
double mean(Array<double,3> immage){
    double res = 0;
    auto data = immage.to_std_vector();
    auto shape = immage.get_shape();
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            for (size_t k = 0; k < shape[2]; ++k) {
                res += data[i * shape[1] * shape[2] + j * shape[2] + k];
            }
        }
    }
    return res/(shape[0]*shape[1]*shape[2]);
}
Array<double,3> image_conversion(Array<Color,2> image){
    auto data = image.to_std_vector();
    auto shape = image.get_shape();
    size_t new_shape[] = {shape[0],shape[1],3};
    auto new_data = std::vector<double>(shape[0] * shape[1] * 3);
    for(size_t i = 0; i < shape[0] * shape[1]; ++i){
        new_data[i*3] = (double) data[i].r;
        new_data[i*3+1] = (double) data[i].g;
        new_data[i*3+2] = (double) data[i].b;
    }
    return Array<double,3>(new_data, new_shape);
}
double ssim(Array<Color,2> x,Array<Color,2> y){
    Array<double,3> x_double = image_conversion(x);
    Array<double,3> y_double = image_conversion(y);
    double k_1 = 0.01;
    double k_2 = 0.03;
    double L = 1;
    double c_1 = (k_1 * L) * (k_1 * L);
    double c_2 = (k_2 * L) * (k_2 * L);
    double c_3 = c_2 / 2;
    /*double mux = mean(x);*/
    double mux = mean(x_double);
    /*double muy = mean(y);*/
    double muy = mean(y_double);
    /*Array<Color,2> x_minus_mux = (x-mean(x));*/
    Array<double,3> x_minus_mux = (x_double-mux);
    /*Array<Color,2> y_minus_muy = (y-mean(y));*/
    Array<double,3> y_minus_muy = (y_double-muy);
    /*double sigma_x_squared = mean(x_minus_mux.apply<Color>([](Color c){return c*c;}));*/
    /*double sigma_x_squared = mean(x_minus_mux.apply<double>([](Color c){return ((double)(c.r*c.r)+(double)(c.g*c.g)+(double)(c.b*c.b))/(3.);}));*/
    double sigma_x_squared = mean(x_minus_mux.apply<double>([](double c){return c*c;}));
    double sigma_x = std::sqrt(sigma_x_squared);
    /*double sigma_y_squared = mean(y_minus_muy.apply<Color>([](Color c){return c*c;}));*/
    /*double sigma_y_squared = mean(y_minus_muy.apply<double>([](Color c){return ((double)(c.r*c.r)+(double)(c.g*c.g)+(double)(c.b*c.b))/(3.);}));*/
    double sigma_y_squared = mean(y_minus_muy.apply<double>([](double c){return c*c;}));
    double sigma_y = std::sqrt(sigma_y_squared);
    double cov_xy = mean(
                            (x_minus_mux*y_minus_muy)
                          );
    double l = (2*mux*muy + c_1)/(mux*mux + muy*muy + c_1);
    double c = (2*sigma_x*sigma_y + c_2)/(sigma_x_squared + sigma_y_squared + c_2);
    double s = (cov_xy + c_3)/(sigma_x*sigma_y + c_3);
    return l*c*s;
}
class RayTracer {
private:
    size_t width, height;
    std::vector<SceneObject> scene_objects;
    int max_bounces;

public:
    RayTracer(int w, int h) : width(w), height(h), max_bounces(7) {}/*max_bounces(7) {}*/

    void add_object(Vector3D position, float radius, Color color, Material material, float emission) {
        scene_objects.push_back(SceneObject(position, radius, color, material, emission));
    }
    /*define the image normalisation*/
    /*def image_normalisation(image):
        logtransformfactor = 50
        if logtransformfactor != 0:
            img = image * logtransformfactor
            img = np.log(img + 1)
        img = img / np.max(img)
        return img
    */
    Array<Color,2> image_normalisation(const Array<Color,2> image){
        /*create a vector to then initialise the image*/
        auto vcol = std::vector<Color>(height * width);
        /*initialise the image*/
        for(int i = 0; i < height * width; ++i) {
            vcol[i] = Color(0, 0, 0);
        }
        size_t shape[] = {height, width};
        Array<Color,2> res(vcol, shape);
        float logtransformfactor = 50;
        if (logtransformfactor != 0) {
            res = image* logtransformfactor ;
            Array<Color, 2> res2 = res.apply<Color>([](Color c) {return Color(std::log(c.r + 1),std::log(c.g + 1),std::log(c.b + 1));});
            res2 = res2 / max_image(res2);
            return res2;
        }
        else{
            res = image.copy();
            res = res / max_image(res);
            return res;
        }

    }

    Vector3D calculate_intersection(Vector3D ray_origin, Vector3D ray_direction, SceneObject obj) {
        Vector3D oc = ray_origin - obj.position;
        float a = ray_direction.dot(ray_direction);
        float b = 2.0 * oc.dot(ray_direction);
        float c = oc.dot(oc) - obj.radius * obj.radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant > 0) {
            float root1 = (-b - sqrt(discriminant)) / (2 * a);
            float root2 = (-b + sqrt(discriminant)) / (2 * a);
            if (root1 > 0) {
                return ray_origin + ray_direction * root1;
            } else if (root2 > 0) {
                return ray_origin + ray_direction * root2;
            }
        }

        return Vector3D(); // Return zero vector when no intersection is found
    }

    Vector3D calculate_surface_normal(SceneObject obj, Vector3D hit_point) {
        return (hit_point - obj.position).normalize();
    }

    Vector3D reflect(Vector3D incident, Vector3D normal) {
        return incident - normal * (2 * incident.dot(normal));
    }

    Vector3D random_rotate(Vector3D direction, float angle_std, std::default_random_engine& generator) {
        std::normal_distribution<float> distribution(0.0, angle_std);

        /*float phi = distribution(generator);
        float theta = distribution(generator);

        float cos_phi = std::cos(phi);
        float sin_phi = std::sin(phi);
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        Vector3D new_direction(
            direction.x * cos_phi + sin_phi * (direction.y * cos_theta + direction.z * sin_theta),
            direction.y * cos_phi - sin_phi * (direction.x * cos_theta - direction.z * sin_theta),
            direction.z * cos_theta - direction.y * sin_theta
        );
        Vector3D new_direction = (direction + Vector3D(distribution(generator), distribution(generator), distribution(generator))).normalize();*/
        float initial_angle_theta = std::atan2(direction.y, direction.x);
        float initial_angle_phi = std::acos(direction.z);
        float angle_phi = initial_angle_phi+distribution(generator)/2;
        float angle_theta = initial_angle_theta+distribution(generator);
        float cos_phi = std::cos(angle_phi);
        float sin_phi = std::sin(angle_phi);
        float cos_theta = std::cos(angle_theta);
        float sin_theta = std::sin(angle_theta);
        Vector3D new_direction(
            sin_phi * cos_theta,
            sin_phi * sin_theta,
            cos_phi
        );
        return new_direction;
    }
    Color trace_ray(Vector3D ray_origin, Vector3D ray_direction, std::default_random_engine& generator) {
        Color ray_color = Color(0, 0, 0);
        Color energy(1.0f, 1.0f, 1.0f);

        for (int loop = 0; loop < max_bounces; ++loop) {
            std::tuple<float, const SceneObject*, Vector3D> closest_hit;
            for (const auto& obj : scene_objects) {
                Vector3D hit_point = calculate_intersection(ray_origin, ray_direction, obj);
                if (hit_point.magnitude() > 0) {
                    float distance_to_hit = (hit_point - ray_origin).magnitude();
                    if (std::get<0>(closest_hit) == 0 || distance_to_hit < std::get<0>(closest_hit)) {
                        closest_hit = std::make_tuple(distance_to_hit, &obj, hit_point);
                    }
                }
            }

            if (std::get<0>(closest_hit) == 0) {
                break;
            }

            float distance_to_hit = std::get<0>(closest_hit);
            const SceneObject* hit_obj = std::get<1>(closest_hit);
            Vector3D hit_point = std::get<2>(closest_hit);
            Material material = hit_obj->material;

            Vector3D normal = calculate_surface_normal(*hit_obj, hit_point);

            Color lighting_color = Color(0, 0, 0);
            if (hit_obj->emission > 0) {
                Color light_intensity = hit_obj->color * hit_obj->emission;
                lighting_color += light_intensity;
            }


            /*if(hit_point.z>0){*/
                /*print hit point*/
                /*std::cout << "hit point: " << hit_point.x << " " << hit_point.y << " " << hit_point.z << std::endl;
                std::cout << "initial energy" << energy.r << " " << energy.g << " " << energy.b << std::endl;
                std::cout << "lighting color" << lighting_color.r << " " << lighting_color.g << " " << lighting_color.b << std::endl;
                std::cout << "initial ray color" << ray_color.r << " " << ray_color.g << " " << ray_color.b << std::endl;
                std::cout << "object color" << hit_obj->color.r << " " << hit_obj->color.g << " " << hit_obj->color.b << std::endl;
                std::cout << "object emission" << hit_obj->emission << std::endl;
                std::cout << "object reflection" << hit_obj->material.reflection_coefficient << std::endl;
            }*/
            ray_color += energy * lighting_color;
            energy *= hit_obj->color;
            /*if (hit_point.z > 0) {*/
                /*print hit point*/
                /*std::cout << "energy after lighting" << energy.r << " " << energy.g << " " << energy.b << std::endl;
            }*/
            energy *= hit_obj->material.reflection_coefficient;
            /*if(hit_point.z>0){*/
                /*print hit point*/
                /*std::cout << "final energy" << energy.r << " " << energy.g << " " << energy.b << std::endl;
                std::cout << "final ray color" << ray_color.r << " " << ray_color.g << " " << ray_color.b << std::endl;

            }*/
            
            /*if not the last bounce then bounce*/
            if(loop < max_bounces - 1) {
                Vector3D reflection_ray_origin = hit_point + normal * 0.001f;
                Vector3D reflection_ray_direction = reflect(ray_direction, normal);

                if (material.diffuse_coefficient != 0) {
                    Vector3D new_reflection_ray_direction = random_rotate(reflection_ray_direction, material.diffuse_coefficient, generator);
                    int k = 30;
                    while (new_reflection_ray_direction.dot(normal) <= 0) {
                        new_reflection_ray_direction = random_rotate(reflection_ray_direction, material.diffuse_coefficient, generator);
                        k -= 1;
                        if (k == 0) {
                            /*brake and return raycolor*/
                            
                            return ray_color;
                            /*raise an error*/
                            /*int a  = generator();
                            int b = generator();
                            throw std::runtime_error(std::string("k == 0")+std::to_string(a)+std::to_string(b));*/
                        }
                    }
                    reflection_ray_direction = new_reflection_ray_direction;
                }

                ray_origin = reflection_ray_origin;
                ray_direction = reflection_ray_direction;
            }
            /*std::cout << "loop: " << loop << " " << ray_color.r << " " << ray_color.g << " " << ray_color.b << std::endl;*/
        }
        return ray_color;
    }
    size_t bouncing(float diffuse_coefficient){
        /*bigger is the coefficient more bounce we will do*/
        /*1+log(1+coef)*/
        return (size_t)(1+std::log(1+diffuse_coefficient));
        /*return 1;*/
    }
    Color trace_ray_2(Vector3D ray_origin, Vector3D ray_direction, std::default_random_engine& generator){
        /*same as trace_ray_2 but we will do more bounce according to bounce*/
        /*store the tracing to do in a queue*/
        std::queue<std::tuple<Vector3D,Vector3D,Color,size_t>> tasks;
        /*store the firs task in it*/
        tasks.push(std::make_tuple(ray_origin,ray_direction,Color(1,1,1),max_bounces));
        Color ray_color = Color(0, 0, 0);
        /*while there is task to do*/
        while (!tasks.empty()) {
            /*get the task*/
            std::tuple<Vector3D,Vector3D,Color,size_t> task = tasks.front();
            tasks.pop();
            /*get the task parameters*/
            Vector3D ray_origin = std::get<0>(task);
            Vector3D ray_direction = std::get<1>(task);
            Color energy = std::get<2>(task);
            size_t bounce = std::get<3>(task);
            
            /*compute intersection*/
            std::tuple<float, const SceneObject*, Vector3D> closest_hit;
            for (const auto& obj : scene_objects) {
                Vector3D hit_point = calculate_intersection(ray_origin, ray_direction, obj);
                if (hit_point.magnitude() > 0) {
                    float distance_to_hit = (hit_point - ray_origin).magnitude();
                    if (std::get<0>(closest_hit) == 0 || distance_to_hit < std::get<0>(closest_hit)) {
                        closest_hit = std::make_tuple(distance_to_hit, &obj, hit_point);
                    }
                }
            }
            /*if there is no intersection*/
            if (std::get<0>(closest_hit) == 0) {
                continue;
            }

            /*get the intersection parameters*/
            float distance_to_hit = std::get<0>(closest_hit);
            const SceneObject* hit_obj = std::get<1>(closest_hit);
            Vector3D hit_point = std::get<2>(closest_hit);
            Material material = hit_obj->material;

            Vector3D normal = calculate_surface_normal(*hit_obj, hit_point);

            Color lighting_color = Color(0, 0, 0);
            if (hit_obj->emission > 0) {
                Color light_intensity = hit_obj->color * hit_obj->emission;
                lighting_color += light_intensity;
            }

            ray_color += energy * lighting_color;
            energy *= hit_obj->color;
            energy *= hit_obj->material.reflection_coefficient;

            /*if not the last bounce then bounce*/
            if(bounce > 1) {
                Vector3D reflection_ray_origin = hit_point + normal * 0.001f;
                Vector3D reflection_ray_direction = reflect(ray_direction, normal);
                size_t bounce_nb = bouncing(material.diffuse_coefficient);
                for(size_t i = 0; i < bounce_nb; ++i){
                    Vector3D new_reflection_ray_direction;
                    if (material.diffuse_coefficient != 0) {
                        new_reflection_ray_direction = random_rotate(reflection_ray_direction, material.diffuse_coefficient, generator);
                        int k = 30;
                        while (new_reflection_ray_direction.dot(normal) <= 0) {
                            new_reflection_ray_direction = random_rotate(reflection_ray_direction, material.diffuse_coefficient, generator);
                            k -= 1;
                            if (k == 0) {
                                break;
                            }
                        }
                        if (k == 0) {
                            /*brake and return raycolor*/
                            continue;
                        }
                    }
                    else{
                        new_reflection_ray_direction = reflection_ray_direction;
                    }
                    /*add the new task to the queue*/
                    tasks.push(std::make_tuple(reflection_ray_origin,new_reflection_ray_direction,energy/bounce_nb,bounce-1));
                }
            }
        }
        return ray_color;
    }
    Color render_pixel(int x, int y, int subsampling, std::default_random_engine& generator) {
        /*uniform distribution*/
        std::uniform_real_distribution<float> distribution(-0.5, 0.5);
        Color ray_color;
        /*set color to 000*/
        ray_color = Color(0, 0, 0);
        for (int sub_samp = 0; sub_samp < subsampling; ++sub_samp) {
            Vector3D ray_direction(
                static_cast<float>(((x + static_cast<float>(distribution(generator)))+0.5f) / width) - 0.5f,
                static_cast<float>(((y + static_cast<float>(distribution(generator)))+0.5f) / height) - 0.5f,
                1.0f
            );
            auto ray_res = trace_ray_2(Vector3D(0, 0, 0), ray_direction.normalize(), generator);
            ray_color += ray_res;
        }
        /*return ray_color;*/
        return ray_color;
    }
    /*usage :
        for (int i = 0; i < nb_threads; ++i) {
                        threads[i] = std::thread(&RayTracer::render_pixel_thread, this, std::ref(tasks[i]), std::ref(results[i]));
                    }
    */
    /*render pixel thread with a que of tax an mutex to acsess it*/
    void render_pixel_thread_2(std::queue<std::pair<std::pair<int,int>,int>>& tasks, std::vector<std::pair<std::pair<int,int>,Color>>& results, std::mutex& mtx, std::default_random_engine& generator){
        while (true) {
            /*lock the mutex*/
            mtx.lock();
            /*acsess the queue*/
            if (tasks.empty()) {
                mtx.unlock();
                break;
            }
            std::pair<std::pair<int,int>,int> task = tasks.front();
            tasks.pop();
            /*realease the mutex*/
            mtx.unlock();
            /*deal  with the task*/
            results.push_back(std::make_pair(task.first,render_pixel(task.first.first, task.first.second, task.second,generator)));
        }
    }
    Array<Color,2> render(std::vector<double>& perf_hist, std::vector<float>& times_hist, std::vector<float>& sample_hist,Array<Color,2> compare, bool bandit = false, bool compare_bool = false, size_t samples_per_pixel = 1000, size_t subsampling = 1) {
        std::clock_t start_time = std::clock();
        /*create a vector to then initialise the image*/
        auto vcol = std::vector<Color>(height * width);
        /*initialise the image*/
        for(int i = 0; i < height * width; ++i) {
            vcol[i] = Color(0, 0, 0);
        }
        size_t shape[] = {height, width};
        Array<Color,2> image(vcol, shape);
        Array<float,2> times(shape);
        Array<float,2> losts(shape);
        times.fill(0);
        losts.fill(0);
        float nu =  std::sqrt(std::log(width*height)/(width*height*samples_per_pixel*subsampling));
        nu *= 1;
        std::cout << "nu: " << nu << std::endl;
        /*create probas*/
        Array<float,2> probas(shape);
        /*initialize probas*/
        if (bandit) {
            probas.fill(1.0f / (width * height));
        }

        for (int smp = 0; smp < samples_per_pixel; ++smp) {
            std::cout << (int)(100*((double)(smp))/((double)samples_per_pixel)) << "%" << std::endl;
            if (smp != 0) {
                float tm = (std::clock() - start_time) / (float)CLOCKS_PER_SEC;
                std::cout << "start since : " << (int) (tm / 60) << "min" << (int)(tm - 60 * (int)(tm / 60)) << "s" << std::endl;
                std::cout << "estimated rendering time " << (int) (samples_per_pixel * (tm / (smp)) / 60) << "min" << (int)(samples_per_pixel * (tm / (smp)) - 60 * (int)(samples_per_pixel * (tm / (smp)) / 60)) << "s" << std::endl;
                std::cout << "estimated time left " << (int)((samples_per_pixel - smp) * (tm / (smp)) / 60) << "min" << (int)((samples_per_pixel - smp) * (tm / (smp)) - 60 * (int)((samples_per_pixel - smp) * (tm / (smp)) / 60)) << "s" << std::endl;
            }
            int jbs = -1;
            if (!bandit || smp == 0) {
                bool parallel = true;
                std::vector<Color> newrays;
                /*create generator*/
                std::default_random_engine generator(std::time(nullptr));
                if (!parallel){
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            newrays.push_back(render_pixel(x, y, subsampling, generator));
                            /*std::cout << "pixel: " << x << " " << y << " : "<< newrays.back().r << " " << newrays.back().g << " " << newrays.back().b << std::endl;*/
                        }
                    }
                }
                else{
                    /*threads with queues*/
                    int nb_threads = 30;
                    /*std::cout << "creating vectors" << std::endl;*/
                    std::queue<std::pair<std::pair<int,int>,int>> tasks;
                    std::vector<std::vector<std::pair<std::pair<int,int>,Color>>> results(nb_threads);
                    std::mutex mtx;
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            tasks.push(std::make_pair(std::make_pair(x,y),subsampling));
                        }
                    }
                    std::vector<std::thread> threads(nb_threads);
                    /*create multiple generators*/
                    std::vector<std::default_random_engine> generators(nb_threads);
                    int tim = std::time(nullptr);
                    for (int i = 0; i < nb_threads; ++i) {
                        generators[i] = std::default_random_engine(i+tim);
                    }
                    /*std::cout << "create threads" << std::endl;*/
                    for (int i = 0; i < nb_threads; ++i) {
                        threads[i] = std::thread(&RayTracer::render_pixel_thread_2, this, std::ref(tasks), std::ref(results[i]), std::ref(mtx), std::ref(generators[i]));
                    }
                    /*std::cout << "join threads" << std::endl;*/
                    for (int i = 0; i < nb_threads; ++i) {
                        threads[i].join();
                        /*std::cout << "thread " << i << " joined" << std::endl;*/
                    }
                    /*std::cout << "get results" << std::endl;*/
                    std::vector<int> used_times(nb_threads,0);
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            /*find which process did this task*/
                            /*std::cout << "x: " << x << " y: " << y << std::endl;*/
                            int index = -1;
                            for (int i = 0; i < nb_threads; ++i) {
                                if (used_times[i] < results[i].size()) {
                                    /*std::cout << "i: " << i << " used_times[i]: " << used_times[i] << " results[i][used_times[i]].first: " << results[i][used_times[i]].first.first << " " << results[i][used_times[i]].first.second << std::endl;*/
                                    if (results[i][used_times[i]].first == std::make_pair(x,y)) {
                                        index = i;
                                        break;
                                    }
                                }
                            }
                            if (index == -1) {
                                std::cout << "error: no process did this task" << std::endl;
                            }
                            newrays.push_back(results[index][used_times[index]].second);
                            ++used_times[index];
                        }
                    }

                }
                auto new_to_add = Array<Color,2>(newrays, shape);
                /*std::cout << "image to add : "<< std::endl;
                for (int y = 0; y < height; ++y) {
                    std::cout << "line " << y << ": ";
                    for (int x = 0; x < width; ++x) {
                        std::cout << new_to_add.to_std_vector()[y * width + x].r << " " << new_to_add.to_std_vector()[y * width + x].g << " " << new_to_add.to_std_vector()[y * width + x].b << " ";
                    }
                    std::cout << std::endl;
                }*/
                image += new_to_add;
                /*std::cout<< "image 25 25 : "<< image.to_std_vector()[25*50+25].r << " " << image.to_std_vector()[25*50+25].g << " " << image.to_std_vector()[25*50+25].b << std::endl;*/
                
                times += Array<float,2>(std::vector<float>(height * width, subsampling), shape);
            } else {
                /*std::cout << "random generation" << std::endl;*/
                auto pixels = probas.list_coord_freq2(height * width*subsampling);
                /*std::cout << "random generation done" << std::endl;*/
                std::vector<Color> newrays;
                /*create generator*/
                bool parallel = true;
                Array<Color,2> m(shape);
                Array<float,2> tms(std::vector<float>(height * width, 0), shape);
                if (!parallel){
                    std::default_random_engine generator(std::time(nullptr));
                    for (const auto& element : pixels) {
                        int corx,cordy;
                        corx = element.first % width;
                        cordy = element.first / width;
                        /*std::cout << element.first << std::endl;*/
                        m.set(element.first, render_pixel(corx, cordy, element.second,generator));
                        tms.set(element.first, element.second);
                    }
                }
                else{
                    /*std::cout << "parallel" << std::endl;*/
                    /*threads with queues*/
                    int nb_threads = 30;
                    /*std::cout << "creating vectors" << std::endl;*/
                    std::queue<std::pair<std::pair<int,int>,int>> tasks;
                    std::vector<std::vector<std::pair<std::pair<int,int>,Color>>> results(nb_threads);
                    std::mutex mtx;
                    for (const auto& element : pixels) {
                        tasks.push(std::make_pair(std::make_pair(element.first % width,element.first / width),element.second));
                        tms.set(element.first, element.second);
                    }
                    std::vector<std::thread> threads(nb_threads);
                    /*create multiple generators*/
                    std::vector<std::default_random_engine> generators(nb_threads);
                    int tim = std::time(nullptr);
                    for (int i = 0; i < nb_threads; ++i) {
                        generators[i] = std::default_random_engine(i+tim);
                    }
                    /*std::cout << "create threads" << std::endl;*/
                    for (int i = 0; i < nb_threads; ++i) {
                        threads[i] = std::thread(&RayTracer::render_pixel_thread_2, this, std::ref(tasks), std::ref(results[i]), std::ref(mtx), std::ref(generators[i]));
                    }
                    /*std::cout << "join threads" << std::endl;*/
                    for (int i = 0; i < nb_threads; ++i) {
                        threads[i].join();
                        /*std::cout << "thread " << i << " joined" << std::endl;*/
                    }
                    /*std::cout << "get results" << std::endl;*/
                    std::vector<int> used_times(nb_threads,0);
                    for (const auto& element : pixels) {
                        /*find which process did this task*/
                        /*std::cout << "x: " << x << " y: " << y << std::endl;*/
                        int index = -1;
                        for (int i = 0; i < nb_threads; ++i) {
                            if (used_times[i] < results[i].size()) {
                                /*std::cout << "i: " << i << " used_times[i]: " << used_times[i] << " results[i][used_times[i]].first: " << results[i][used_times[i]].first.first << " " << results[i][used_times[i]].first.second << std::endl;*/
                                if (results[i][used_times[i]].first == std::make_pair((int)(element.first % width),(int)(element.first / width))) {
                                    index = i;
                                    break;
                                }
                            }
                        }
                        if (index == -1) {
                            std::cout << "error: no process did this task" << std::endl;
                        }
                        else{
                            m.set(element.first, results[index][used_times[index]].second);
                            ++used_times[index];
                        }
                    }
                    /*std::cout << "end parallel" << std::endl;*/
                }
                image += m;
                times += tms;
                /*losts +=subsampling*(
                                    np.sum(
                                        (tms>0)*(
                                            1-(np.abs(
                                                image_normalisation(
                                                    (m+image)/(times + tms+0.0001)
                                                )-image_normalisation(
                                                    image/(times+0.0001)
                                                )
                                            )/(tms+0.000001))
                                        )/3
                                        ,axis = 2
                                    )

                                )*/
                Array<Color,2> color_diff = (image_normalisation((m + image) / (times + tms + 0.0001)) - image_normalisation(image / (times + 0.0001))).apply<Color>([](Color c) {return Color(std::abs(c.r), std::abs(c.g), std::abs(c.b)); });
                
                Array<float,2> color_diff_sum = color_diff.apply<float>([](Color c) {return 1- ((c.r + c.g + c.b) / 3); });
                /*it is too near to 1*/
                /*let make it more beetwin 0 and 1*/
                /*Array<float,2> color_diff_sum_bis = color_diff_sum.apply<float>([](float f) {if(f<0.99){return f/(99.f);}else{return (f-(0.99f))*(99.f);}});*/
                Array<float,2> color_diff_sum_bis =((color_diff_sum-color_diff_sum.min())/(color_diff_sum.max()-color_diff_sum.min()));
                if(false){
                    /*save color diff sum*/
                    save_image(to_255((color_diff_sum-color_diff_sum.min())/(color_diff_sum.max()-color_diff_sum.min())), "Monte Carlo Ray Tracing color diff sum.txt");
                    /*save old immage*/
                    save_image(to_255(image_normalisation(image / (times + 0.0001))), "Monte Carlo Ray Tracing old image.txt");
                    /*save new times*/
                    std::cout << "tms: " << tms.min() << "to" << tms.max() << std::endl;
                    save_image(to_255(tms/tms.max()), "Monte Carlo Ray Tracing new times.txt");
                    /*save dolor diff sum bis*/
                    save_image(to_255(color_diff_sum_bis), "Monte Carlo Ray Tracing color diff sum bis.txt");
                }
                Array<float,2> tms_pos = tms;/*tms.apply<float>([](float f) {if(f>0){return 1.f;}else{return 0.f;}});*/
                Array<float,2> losts_add = (tms_pos * color_diff_sum_bis); /* * subsampling;*/
                losts += losts_add;
                losts-=losts.min();
                /*std::cout << "losts: " << losts.min() << "to" << losts.max() << std::endl;*/
                Array<float,2> exps = (losts*(-nu)).apply<float>(std::exp);
                /*std::cout << "exps: " << exps.min() << "to" << exps.max() << std::endl;*/ 
                float sm = exps.sum();
                /*std::cout << "sm: " << sm << std::endl;*/
                probas = exps / sm;
                /*std::cout << "probas: " << probas.min() << "to" << probas.max() << std::endl;*/
            /*std::cout << image.get_shape() << std::endl;*/
            }
            if (compare_bool) {
                times_hist.push_back((std::clock() - start_time) / (float)CLOCKS_PER_SEC);
                /*perf_hist.push_back(
                    (
                        (
                            image_normalisation(compare) - image_normalisation(image / (times + 0.0001))
                            ).apply<float>([](Color x) {return x.r * x.r  +  x.g * x.g  +  x.b * x.b; })
                        ).sum()
                    );*/
                perf_hist.push_back(1.-ssim(compare, image / (times)));
                sample_hist.push_back(times.sum());
            }
            /*show images*/
            /*cv::Mat image_cv = to_cv_mat(image);
            cv::imshow("Monte Carlo Ray Tracing", image_cv);
            cv::waitKey(0);*/
            /*show bandits*/
            /*if (bandit) {
                cv::Mat probas_cv = to_cv_mat(probas);
                cv::imshow("Monte Carlo Ray Tracing probas", probas_cv);
                cv::waitKey(0);
            }
            */
            /*save images*/
            /*print that we are saving images*/
            if(false){std::cout << "saving images" << std::endl;
                save_image(to_255(image_normalisation(image/times)), "Monte Carlo Ray Tracing.txt");
                if (bandit) {
                    save_image(to_255(probas/probas.max()), "Monte Carlo Ray Tracing probas.txt");
                    /*save losts*/
                    save_image(to_255(losts / losts.max()), "Monte Carlo Ray Tracing losts.txt");
                    /*save times*/
                    save_image(to_255(times / times.max()), "Monte Carlo Ray Tracing times.txt");
                }
            }
            /*show image in terminal*/
            /*std::cout << "image:" << std::endl;
            for (int y = 0; y < height; ++y) {
                std::cout << "line " << y << ": ";
                for (int x = 0; x < width; ++x) {
                    std::cout << image.to_std_vector()[y * width + x].r << " " << image.to_std_vector()[y * width + x].g << " " << image.to_std_vector()[y * width + x].b << " ";
                }
                std::cout << std::endl;
            }*/
        }
        save_image(to_255(image_normalisation(image / (times + 0.0001))), "Monte Carlo Ray Tracing.txt");
        if (bandit) {
            save_image(to_255(probas/probas.max()), "Monte Carlo Ray Tracing probas.txt");
            /*save losts*/
            save_image(to_255(losts / losts.max()), "Monte Carlo Ray Tracing losts.txt");
            /*save times*/
            save_image(to_255(times / times.max()), "Monte Carlo Ray Tracing times.txt");
        }
        return image / (times + 0.0001);
    }



};

int main() {
    size_t width =50;
    size_t height = 50;

    RayTracer ray_tracer(width, height);
    /*left- right+/ up- down+/ front+ back-*/
    ray_tracer.add_object(Vector3D(0, -1, -1), 0.9f, Color(1, 0.5, 0.0), Material(0, 0), 1);
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
    Array<Color,2> image = ray_tracer.render(perf_hist_null, times_hist_null, sample_hist_null, compare_null, false, false, 50, 10000);
    /*save the result*/
    save_image(ray_tracer.image_normalisation(image), "Monte Carlo Ray Tracing reference.txt");
    /*create an immage with classic method and save metrics*/
    std::vector<double> perf_hist_classic;
    std::vector<float> times_hist_classic;
    std::vector<float> sample_hist_classic;
    Array<Color,2> image_classic = ray_tracer.render(perf_hist_classic, times_hist_classic, sample_hist_classic, image, false, true, 50, 500);
    /*save the result*/
    save_image(ray_tracer.image_normalisation(image_classic), "Monte Carlo Ray Tracing classic.txt");
    /*save metrics*/
    save_metrics(perf_hist_classic, times_hist_classic, sample_hist_classic, "Monte Carlo Ray Tracing classic metrics.txt");
    /*create an immage with bandit method and save metrics*/
    std::vector<double> perf_hist_bandit;
    std::vector<float> times_hist_bandit;
    std::vector<float> sample_hist_bandit;
    Array<Color,2> image_bandit = ray_tracer.render(perf_hist_bandit, times_hist_bandit, sample_hist_bandit, image, true, true, 50, 500);
    /*save the result*/
    save_image(ray_tracer.image_normalisation(image_bandit), "Monte Carlo Ray Tracing bandit.txt");
    /*save metrics*/
    save_metrics(perf_hist_bandit, times_hist_bandit, sample_hist_bandit, "Monte Carlo Ray Tracing bandit metrics.txt");
    return 0;
}




