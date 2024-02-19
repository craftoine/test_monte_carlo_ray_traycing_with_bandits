"""void save_metrics(std::vector<float> perf_hist,std::vector<float> times_hist,std::vector<float> sample_hist,std::string filename){
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
    std::cout << "file created" << std::endl;
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
}"""
import numpy as np
import matplotlib.pyplot as plt
def import_metrics(path):
    """import metrics from path"""
    with open(path, 'r') as f:
        lines = f.readlines()
    perf_hist = np.abs([float(i) for i in lines[1].split()])
    times_hist = [float(i) for i in lines[3].split()]
    sample_hist = [float(i) for i in lines[5].split()]
    return perf_hist, times_hist, sample_hist
"""
    save_metrics(perf_hist_classic, times_hist_classic, sample_hist_classic, "Monte Carlo Ray Tracing classic metrics.txt");
    save_metrics(perf_hist_bandit, times_hist_bandit, sample_hist_bandit, "Monte Carlo Ray Tracing bandit metrics.txt");
"""
def show_metrics():
    """show metrics"""
    """ semi log y axis for perf"""
    perf_hist_classic, times_hist_classic, sample_hist_classic = import_metrics("Monte Carlo Ray Tracing classic metrics.txt")
    perf_hist_bandit, times_hist_bandit, sample_hist_bandit = import_metrics("Monte Carlo Ray Tracing bandit metrics.txt")
    """ plot the perf in fuction of time"""
    if len(times_hist_classic)>1:
        plt.plot(times_hist_classic, perf_hist_classic, label='classic', color='red')
    else:
        #plot as a single point x
        plt.scatter(times_hist_classic, perf_hist_classic, label='classic', marker='x', color='red')
    if len(times_hist_bandit)>1:
        plt.plot(times_hist_bandit, perf_hist_bandit, label='bandit',color='blue')
    else:
        #plot as a single point x
        plt.scatter(times_hist_bandit, perf_hist_bandit, label='bandit', marker='x', color='blue')
    plt.yscale('log')
    plt.xlabel('time')
    plt.ylabel('immage distance to reference (1-ssim)')
    plt.legend()
    plt.show()
    """ plot the perf in fuction of sample"""
    if len(times_hist_classic)>1:
        plt.plot(sample_hist_classic, perf_hist_classic, label='classic', color='red')
    else:
        #plot as a single point x
        plt.scatter(sample_hist_classic, perf_hist_classic, label='classic', marker='x', color='red')
    if len(times_hist_bandit)>1:
        plt.plot(sample_hist_bandit, perf_hist_bandit, label='bandit',color='blue')
    else:
        #plot as a single point x
        plt.scatter(sample_hist_bandit, perf_hist_bandit, label='bandit', marker='x', color='blue')
    plt.yscale('log')
    plt.xlabel('sample')
    plt.ylabel('immage distance to reference (1-ssim)')
    plt.legend()
    plt.show()
    """ plot the sample in function of time"""
    if len(times_hist_classic)>1:
        plt.plot(times_hist_classic, sample_hist_classic, label='classic', color='red')
    else:
        #plot as a single point x
        plt.scatter(times_hist_classic, sample_hist_classic, label='classic', marker='x', color='red')
    if len(times_hist_bandit)>1:
        plt.plot(times_hist_bandit, sample_hist_bandit, label='bandit',color='blue')
    else:
        #plot as a single point x
        plt.scatter(times_hist_bandit, sample_hist_bandit, label='bandit', marker='x', color='blue')
    plt.xlabel('time')
    plt.ylabel('sample')
    plt.legend()
    plt.show()
show_metrics()