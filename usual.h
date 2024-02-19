/*array of color to 0-255 array*/
/*for self as Array<color, 2>*/
Array<unsigned char, 3> to_255(Array<Color, 2> a){
    //std::cout << "to_255 from color" << std::endl;
    auto data = a.to_std_vector();
    auto shape = a.get_shape();
    /*vector shape[0],shape[1],3*/
    size_t new_shape[] = {shape[0],shape[1],3};
    //std::cout << "shape:" << shape[0] << " " << shape[1] << std::endl;
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
    //std::cout << "to_255 from color end" << std::endl;
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
float mean(Array<float,2> immage){
    float res = 0;
    auto data = immage.to_std_vector();
    auto shape = immage.get_shape();
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            res += data[i * shape[1] + j];
        }
    }
    return res/(shape[0]*shape[1]);

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
    std::cout << "mux:" << mux << " muy:" << muy << std::endl;
    /*Array<Color,2> x_minus_mux = (x-mean(x));*/
    Array<double,3> x_minus_mux = (x_double-mux);
    /*Array<Color,2> y_minus_muy = (y-mean(y));*/
    Array<double,3> y_minus_muy = (y_double-muy);
    double sigma_x_squared = mean(x_minus_mux.apply<double>([](double c){return c*c;}));
    double sigma_x = std::sqrt(sigma_x_squared);
    double sigma_y_squared = mean(y_minus_muy.apply<double>([](double c){return c*c;}));
    double sigma_y = std::sqrt(sigma_y_squared);
    double cov_xy = mean(
                            (x_minus_mux*y_minus_muy)
                          );
    //std::cout << "sigma_x_squared:" << sigma_x_squared << " sigma_x:" << sigma_x << " sigma_y_squared:" << sigma_y_squared << " sigma_y:" << sigma_y << " cov_xy:" << cov_xy << std::endl;
    double l = (2*mux*muy + c_1)/(mux*mux + muy*muy + c_1);
    double c = (2*sigma_x*sigma_y + c_2)/(sigma_x_squared + sigma_y_squared + c_2);
    double s = (cov_xy + c_3)/(sigma_x*sigma_y + c_3);
    //std::cout << "l:" << l << " c:" << c << " s:" << s << std::endl;
    return l*c*s;
}