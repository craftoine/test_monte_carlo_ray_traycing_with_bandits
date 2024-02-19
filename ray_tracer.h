#include "arrays.h"
#include "objects.h"
#include "usual.h"


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
        //size_t shape[] = {height, width};
        //Array<Color,2> res(vcol, shape);
        //float logtransformfactor = 50;
        //if (logtransformfactor != 0) {
        //    res = image* logtransformfactor ;
        //    Array<Color,2> res2 = res.apply<Color>([](Color c) {return Color(std::log(c.r + 1),std::log(c.g + 1),std::log(c.b + 1));});
        //    
        //    res2 = res2 / max_image(res2);
        //    return res2;
        //}
        //else{
        //    res = image.copy();
        //    res = res / max_image(res);
        //    return res;
        //}

        float gama = 0.3;
        Array<Color,2> res = image.copy();
        float max_r = res.apply<float>([](Color c) {return c.r;}).max();
        float max_g = res.apply<float>([](Color c) {return c.g;}).max();
        float max_b = res.apply<float>([](Color c) {return c.b;}).max();
        float max = std::max(max_r,std::max(max_g,max_b));
        //res = res/max;
        res = res.apply<Color>([](Color c) {return Color(std::pow(c.r, 0.3),std::pow(c.g, 0.3),std::pow(c.b, 0.3));});
        return res;
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
        float wastead_time = 0;
        //std::clock_t start_time = std::clock();
        //use time frio time.h instead of clock so we can use multiple threads
        auto start_time = std::chrono::high_resolution_clock::now();
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
        nu *= 30;//*= 2 * std::sqrt(width*height);???
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
                //float tm = (std::clock() - start_time) / (float)CLOCKS_PER_SEC;
                float tm = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()/1000.;
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
                    
                    bool classic = false;
                    if(classic){
                        /*std::cout << "create threads" << std::endl;*/
                        //clasic
                        for (int i = 0; i < nb_threads; ++i) {
                            threads[i] = std::thread(&RayTracer::render_pixel_thread_2, this, std::ref(tasks), std::ref(results[i]), std::ref(mtx), std::ref(generators[i]));
                        }
                        /*std::cout << "join threads" << std::endl;*/
                        //clasic
                        for (int i = 0; i < nb_threads; ++i) {
                            threads[i].join();
                            /*std::cout << "thread " << i << " joined" << std::endl;*/
                        }
                    }
                    else{
                        //pool
                        for (int i = 0; i < nb_threads; ++i) {
                            pool.add_task(
                                [&tasks, &results, &mtx, &generators, i, this](){
                                    render_pixel_thread_2(tasks, results[i], mtx, generators[i]);
                                }
                            ,i
                            );
                        }
                        //pool
                        pool.wait_all_tasks();
                    }
                    
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
                    bool classic = false;
                    if(classic){
                    /*std::cout << "create threads" << std::endl;*/
                    for (int i = 0; i < nb_threads; ++i) {
                        threads[i] = std::thread(&RayTracer::render_pixel_thread_2, this, std::ref(tasks), std::ref(results[i]), std::ref(mtx), std::ref(generators[i]));
                    }
                    /*std::cout << "join threads" << std::endl;*/
                    for (int i = 0; i < nb_threads; ++i) {
                        threads[i].join();
                        /*std::cout << "thread " << i << " joined" << std::endl;*/
                    }
                    }
                    else{
                        //pool
                        for (int i = 0; i < nb_threads; ++i) {
                            pool.add_task(
                                [&tasks, &results, &mtx, &generators, i, this](){
                                    render_pixel_thread_2(tasks, results[i], mtx, generators[i]);
                                }
                            ,i
                            );
                        }
                        //pool
                        pool.wait_all_tasks();
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
                                                    (m+image)/(times + tms)
                                                )-image_normalisation(
                                                    image/(times)
                                                )
                                            )/(tms))
                                        )/3
                                        ,axis = 2
                                    )

                                )*/
                std::cout<< "m .sum" << m.sum() << std::endl;
                Array<Color,2> im1 = image_normalisation((m + image) / (times + tms));//image_normalisation((m + image) / (times + tms));
                std::cout << "im1: " << im1.min() << "to" << im1.max() << std::endl;
                std::cout << "im1 .sum" << im1.sum() << std::endl;
                Array<Color,2> im2 =  image_normalisation(image / (times));//image_normalisation(image / (times));
                std::cout << "im2: " << im2.min() << "to" << im2.max() << std::endl;
                std::cout << "im2 .sum" << im2.sum() << std::endl;
                Array<Color,2> color_diff = ( im1- im2).apply<Color>([](Color c) {return Color(std::abs(c.r), std::abs(c.g), std::abs(c.b)); }) / (im2.apply<Color>([](Color c) {return Color(c.r + 1.f, c.g + 1.f, c.b + 1.f); }));
                std::cout << "color_diff: " << color_diff.min() << "to" << color_diff.max() << std::endl;
                std::cout << "color diff .sum" << color_diff.sum() << std::endl;
                Array<float,2> color_diff_sum = color_diff.apply<float>([](Color c) {return 1- ((c.r + c.g + c.b) / 3); });
                std::cout << "color_diff_sum: " << color_diff_sum.min() << "to" << color_diff_sum.max() << std::endl;
                std::cout << "color diff sum .sum" << color_diff_sum.sum() << std::endl;
                /*it is too near to 1*/
                /*let make it more beetwin 0 and 1*/
                Array<float,2> color_diff_sum_bis__=((color_diff_sum-color_diff_sum.min())/(color_diff_sum.max()-color_diff_sum.min()));
                std::cout << "color_diff_sum_bis__: " << color_diff_sum_bis__.min() << "to" << color_diff_sum_bis__.max() << std::endl;
                std::cout << "color diff sum bis .sum" << color_diff_sum_bis__.sum() << std::endl;
                //enseure small values are near to 0, big near to one with a sigmoid like 
                Array<float,2> color_diff_sum_bis = color_diff_sum_bis__.apply<float>(
                                                                                        [](float f) {
                                                                                                        return(2/(1+(float)std::exp(25*(1-f))));
                                                                                                    }
                                                                                    );
                std::cout << "color_diff_sum_bis: " << color_diff_sum_bis.min() << "to" << color_diff_sum_bis.max() << std::endl;
                std::cout << "color diff sum bis .sum" << color_diff_sum_bis.sum() << std::endl;
                if(false){
                    /*save color diff sum*/
                    save_image(to_255((color_diff_sum-color_diff_sum.min())/(color_diff_sum.max()-color_diff_sum.min())), "Monte Carlo Ray Tracing color diff sum.txt");
                    /*save old immage*/
                    save_image(to_255(image_normalisation(image / (times))), "Monte Carlo Ray Tracing old image.txt");
                    /*save new times*/
                    std::cout << "tms: " << tms.min() << "to" << tms.max() << std::endl;
                    save_image(to_255(tms/tms.max()), "Monte Carlo Ray Tracing new times.txt");
                    /*save dolor diff sum bis*/
                    save_image(to_255(color_diff_sum_bis), "Monte Carlo Ray Tracing color diff sum bis.txt");
                }
                Array<float,2> tms_pos = tms;/*tms.apply<float>([](float f) {if(f>0){return 1.f;}else{return 0.f;}});*/
                std::cout << "tms_pos: " << tms_pos.min() << "to" << tms_pos.max() << std::endl;
                std::cout << "tms_pos .sum" << tms_pos.sum() << std::endl;
                Array<float,2> losts_add =(tms_pos * color_diff_sum_bis); /* * subsampling;*/
                std::cout << "losts: " << losts.min() << "to" << losts.max() << std::endl;
                std::cout << "losts .sum"<< losts.sum() << std::endl;
                std::cout << "losts_add: " << losts_add.min() << "to" << losts_add.max() << std::endl;
                std::cout << "losts_add .sum"<< losts_add.sum() << std::endl;
                losts += losts_add;
                losts-=losts.min();
                std::cout << "losts: " << losts.min() << "to" << losts.max() << std::endl;
                std::cout << "losts .sum"<< losts.sum() << std::endl;
                Array<float,2> exps = (losts*(-nu)).apply<float>(std::exp);
                std::cout << "exps: " << exps.min() << "to" << exps.max() << std::endl;
                float sm = exps.sum();
                std::cout << "sm: " << sm << std::endl;
                probas = exps / sm;
            /*std::cout << image.get_shape() << std::endl;*/
            }
            if (compare_bool) {
                /*time wasted*/
                //auto start_wasted_time = std::clock();
                auto start_wasted_time = std::chrono::high_resolution_clock::now();
                //times_hist.push_back(((std::clock() - start_time) / (float)CLOCKS_PER_SEC) - wastead_time );
                times_hist.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count()/1000. - wastead_time );
                /*perf_hist.push_back(
                    (
                        (
                            image_normalisation(compare) - image_normalisation(image / (times))
                            ).apply<float>([](Color x) {return x.r * x.r  +  x.g * x.g  +  x.b * x.b; })
                        ).sum()
                    );*/
                double p = 1.-ssim(compare, image / (times));
                std::cout << "p: " << p << std::endl;
                perf_hist.push_back(p);
                sample_hist.push_back(times.sum());
                //wastead_time += (std::clock() - start_wasted_time) / (float)CLOCKS_PER_SEC;
                wastead_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_wasted_time).count()/1000.;
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
        save_image(to_255(image_normalisation(image / (times))), "Monte Carlo Ray Tracing.txt");
        if (bandit) {
            save_image(to_255(probas/probas.max()), "Monte Carlo Ray Tracing probas.txt");
            /*save losts*/
            save_image(to_255(losts / losts.max()), "Monte Carlo Ray Tracing losts.txt");
            /*save times*/
            save_image(to_255(times / times.max()), "Monte Carlo Ray Tracing times.txt");
        }
        if(compare_bool){
            /*if the arrays are of length 1 duplicate the last value*/
            /*if (perf_hist.size() == 1) {
                perf_hist.push_back(perf_hist.back());
                times_hist.push_back(times_hist.back());
                sample_hist.push_back(sample_hist.back());
            }*/
        }
        //std::cout << "cc" << std::endl;
        //Array<Color,2> rt = image / (times);
        //std::cout << "cc2" << std::endl;
        //return rt;
        return image / (times);
    }



};