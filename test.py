from time import sleep,time
import math
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
#plt.ion()
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

       
    def __mul__(self, other):
        #print(type(other),type(other)== Vec3,type(other)==Color)
        if type(other)== Vec3 or type(other)==Color:
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        scalar = other
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    
    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        length = self.length()
        if length > 0:
            return self / length
        return self

class Color(Vec3):
    def __init__(self, r, g, b):
        super().__init__(r, g, b)

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
    if logtransformfactor != 0:
        img =image*logtransformfactor
        img = np.log(img+1)
    img = img/np.max(img)
    return(img)
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
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - obj.radius**2
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            root1 = (-b - math.sqrt(discriminant)) / (2*a)
            root2 = (-b + math.sqrt(discriminant)) / (2*a)
            if root1 > 0:
                return ray_origin + ray_direction * root1  # Return hit point, not root1
            elif root2 > 0:
                return ray_origin + ray_direction * root2  # Return hit point, not root2

        return None  # Return None when no intersection is found

    def calculate_surface_normal(self, obj, hit_point):
        return (hit_point - obj.position).normalize()
    def reflect(self, incident, normal):
        return incident - normal * (2 * incident.dot(normal))

    def random_rotate(self, direction, angle_std):
        # Generate random angles from a normal distribution
        phi = np.random.normal(0, angle_std)
        theta = np.random.normal(0, angle_std)
        
        # Convert spherical coordinates to Cartesian coordinates
        new_direction = Vec3(
            direction.x * np.cos(phi) + np.sin(phi) * (direction.y * np.cos(theta) + direction.z * np.sin(theta)),
            direction.y * np.cos(phi) - np.sin(phi) * (direction.x * np.cos(theta) - direction.z * np.sin(theta)),
            direction.z * np.cos(theta) - direction.y * np.sin(theta)
        )
        return(new_direction)

    def trace_ray(self, ray_origin, ray_direction):
        ray_color = Color(0, 0, 0)
        energy = Color(1,1,1)
        for loop in range(self.max_bounces):
            closest_hit = None
            for obj in self.scene_objects:
                hit_point = self.calculate_intersection(ray_origin, ray_direction, obj)
                if hit_point is not None:
                    distance_to_hit = (hit_point - ray_origin).length()
                    if closest_hit is None or distance_to_hit < closest_hit[0]:
                        closest_hit = (distance_to_hit, obj,hit_point)

            if closest_hit is None:
                break
            distance_to_hit, hit_obj,hit_point = closest_hit
            material = hit_obj.material
            normal = self.calculate_surface_normal(hit_obj, hit_point)

            # Calculate lighting contribution from light-emitting objects
            lighting_color = Color(0, 0, 0)
            if hit_obj.emission > 0:
                light_intensity = hit_obj.color * hit_obj.emission
                lighting_color += light_intensity
            
            # Calculate reflection contribution
            
            reflection_ray_origin = hit_point + normal * 0.001
            reflection_ray_direction = self.reflect(ray_direction, normal)


            # Add normal noise to the bounce direction
            if hit_obj.material.diffuse_coefficient != 0:
                new_reflection_ray_direction = self.random_rotate(reflection_ray_direction, hit_obj.material.diffuse_coefficient)
                while new_reflection_ray_direction.dot(normal) <= 0:
                    new_reflection_ray_direction = self.random_rotate(reflection_ray_direction, hit_obj.material.diffuse_coefficient)
                reflection_ray_direction = new_reflection_ray_direction

            ray_color += energy*lighting_color
            energy *= hit_obj.color
            energy *= hit_obj.material.reflection_coefficient
            ray_origin = reflection_ray_origin
            ray_direction = reflection_ray_direction
        return ray_color

    def render(self,bandit = False,probas = None,compare_bool = False,compare = None,times_hist = None,perf_hist = None,sample_hist = None):
        start_time = time()
        image = np.zeros((self.height, self.width, 3))
        fig, ax = plt.subplots()
        if bandit:
            fig2,ax2 = plt.subplots()
        times= np.zeros((self.height, self.width,3))
        losts = np.zeros((self.height, self.width))
        for smp in range(samples_per_pixel):
            nu = np.sqrt(np.log(width*height)/(width*height*samples_per_pixel*subsampling))
            print(100*(smp)/samples_per_pixel,"%")
            if smp!=0:
                tm = time()-start_time
                print("start since : ", int(tm)//60,"min",int(tm)%60,"s")
                print("estimated rendering time", int(samples_per_pixel*(tm)/(smp))//60,"min",int(samples_per_pixel*(tm)/(smp))%60,"s")
                print("estimated time left", int((samples_per_pixel-smp)*(tm)/(smp))//60,"min",int((samples_per_pixel-smp)*(tm)/(smp))%60,"s")
            jbs = -1#10
            if not (bandit) or smp ==0:
                newrays = Parallel(n_jobs=jbs)(delayed(self.render_pixel)(x, y,subsampling) for y in range(self.height) for x in range(self.width))
                image += np.reshape(
                    np.array(
                        [[newrays[i].x,newrays[i].y,newrays[i].z]for i in range(len(newrays))]
                        )
                    ,
                    (self.height, self.width, 3)
                )
                times +=np.ones((self.height, self.width,3))*subsampling
            else:
                p_flat = probas.ravel()
                ind = np.arange(len(p_flat))

                res = np.column_stack(
                    np.unravel_index(
                        np.random.choice(ind, p=p_flat, size=subsampling*self.width*self.height),
                        probas.shape))
                #print(res)
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
                    m[pixels[i][0],pixels[i][1]] = np.array([newrays[i].x,newrays[i].y,newrays[i].z])
                    tms[pixels[i][0],pixels[i][1]] = [nbs[i]]*3
                losts +=1-(np.sum(
                    np.abs(image_normalisation((m+image)/(times + tms+0.0001))-image_normalisation(image/(times+0.0001)))**2,axis = 2)/3)
                image += m
                times +=tms
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
            
            if bandit:
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
    def render_pixel(self, x, y,subsampling):
        ray_color = Color(0,0,0)
        for sub_samp in range(subsampling):
            ray_direction = Vec3(((x + random.random() - 0.5)/self.width)-0.5, ((y + random.random() - 0.5)/self.width)-0.5, 1)
            ray_color += self.trace_ray(Vec3(0, 0, 0), ray_direction.normalize())
        return ray_color

width=100
height=100
# Setup the scene and render
ray_tracer = RayTracer(width=width, height=height)
"""ray_tracer.add_object(position=Vec3(0, 0, 1), radius=0.8, color=Color(1, 1, 1), material=Material(0.8, 0.2), emission=0)  # Reflective surface
ray_tracer.add_object(position=Vec3(0, 1, 1), radius=0.8, color=Color(0.1, 0.1, 1), material=Material(0.2, 0.8), emission=0)  # Blue diffuse surface
ray_tracer.add_object(position=Vec3(1, 0, 1), radius=0.8, color=Color(1, 0, 0), material=Material(0.5, 0.5), emission=0)   # Red reflective surface
ray_tracer.add_object(position=Vec3(-1, 0, -1), radius=1, color=Color(1, 1, 1), material=Material(0, 0), emission=0.5)
ray_tracer.add_object(position=Vec3(0, -1, -1), radius=0.2, color=Color(0.4, 1, 0.4), material=Material(0, 0), emission=0.8)
#ray_tracer.add_object(position=Vec3(0, 0, 5), radius=1, color=Color(1, 0, 1), material=Material(0, 0), emission=10.0)
"""


#ray_tracer.add_object(position=Vec3(0, 0, 0), radius=9950, color=Color(1, 1, 1), material=Material(0.2, 0), emission=0.05)

ray_tracer.add_object(position=Vec3(0, -1, -1), radius=0.9, color=Color(1, 0.5, 0), material=Material(0, 0), emission=1)
ray_tracer.add_object(position=Vec3(-1,-1, -1), radius=0.5, color=Color(0, 1, 0), material=Material(0, 0), emission=1)
"""ray_tracer.add_object(position=Vec3(-10,0,-10), radius=5, color=Color(1, 1, 1), material=Material(0, 0), emission=1)
ray_tracer.add_object(position=Vec3(-9,0,-9), radius=3, color=Color(0, 0, 0), material=Material(0, 0), emission=0)
"""
ray_tracer.add_object(position=Vec3(0, 1000, 0), radius=998, color=Color(1, 0.2,0.2 ), material=Material(1, 100), emission=0)

ray_tracer.add_object(position=Vec3(0, 0, 1), radius=0.2, color=Color(0.9, 0.9, 1), material=Material(0.9, 0.2), emission=0)
ray_tracer.add_object(position=Vec3(1,-1, 2), radius=1, color=Color(0.9, 0.9, 1), material=Material(0.9,0.2), emission=0)
#ray_tracer.add_object(position=Vec3(3,-3, -3), radius=4, color=Color(1, 1, 1.1), material=Material(0.9,0.2), emission=0)


logtransformfactor = 50





samples_per_pixel = 1000000
subsampling = 1000
samples_per_pixel//=subsampling


image_compare = ray_tracer.render(bandit = False)
np.save("numpy_ref_img_ray", image_compare)

"""image_compare =np.load("numpy_ref_img_ray.npy")"""
samples_per_pixel = 100000
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
samples_per_pixel = 100000
subsampling = 100
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
