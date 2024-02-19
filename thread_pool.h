#include <unistd.h>
#include <functional>
/*create a thread loop*/
void thread_loop(std::vector<std::function<void()>> &tasks,std::vector<std::mutex> &finished, std::vector<std::mutex> &to_do,size_t id,std::vector<bool> &started,std::vector<bool> &kiled)
{
    while (true) {
        /*if (started[id]) {
            std::cout << "waiting for join task : " << id << std::endl;
        }*/
        while (started[id]) {
            usleep(100);
        }
        //std::cout << "waiting task : " << id << std::endl;
        //lock the mutex, it will be unlocked by the main thread wen a task will be added
        to_do[id].lock();
        //unlock the mutex, we don't need it anymore
        to_do[id].unlock();
        //we check if we are kiled
        if (kiled[id]) {
            //std::cout << "killed task : " << id << std::endl;
            return;
        }
        //we did not finish the task yet so we lock the mutex
        finished[id].lock();
        //we started the task so we set the started bool to true
        started[id] = true;
        //we execute the task
        //std::cout << "executing task : " << id << std::endl;
        tasks[id]();
        //std::cout << "executed task : " << id << std::endl;
        //we finished the task so we unlock the mutex
        finished[id].unlock();
        //std::cout << "finished task : " << id << std::endl;
    }
}
/*create a thread pool*/
/*we we do such that each thread will execute exately one task*/
class ThreadPool {
private:
    std::vector<std::thread> threads;
    std::vector<std::function<void()>> tasks;
    std::vector<std::mutex> mtx_do;
    std::vector<std::mutex> mtx_finished;
    std::vector<bool> started;
    std::vector<bool> kiled;

public:
    const size_t nb_threads;
    ThreadPool(size_t nb_threads) : nb_threads(nb_threads) {
        threads = std::vector<std::thread>(nb_threads);
        tasks = std::vector<std::function<void()>>(nb_threads);
        mtx_do = std::vector<std::mutex>(nb_threads);
        mtx_finished = std::vector<std::mutex>(nb_threads);
        started = std::vector<bool>(nb_threads);
        kiled = std::vector<bool>(nb_threads);
        for (size_t i = 0; i < nb_threads; ++i) {
            started[i] = false;
            kiled[i] = false;
            //lock the two mutexs
            mtx_finished[i].lock();
            mtx_do[i].lock();
        }
        for (size_t i = 0; i < nb_threads; ++i) {
            threads[i] = std::thread(thread_loop, std::ref(tasks), std::ref(mtx_finished), std::ref(mtx_do), i, std::ref(started), std::ref(kiled));
        }
    }
    /*add a task to the thread pool*/
    void add_task(std::function<void()> task, size_t id) {
        //std::cout << "adding task : " << id << std::endl;
        //the do mutex is already locked
        //the finished mutex is already locked
        //we add the task to the task vector
        tasks[id] = task;
        //we unlock the finished mutex
        mtx_finished[id].unlock();
        //we unlock the do mutex
        mtx_do[id].unlock();
        //std::cout << "added task : " << id << std::endl;
    }
    void add_all_tasks(std::vector<std::function<void()>> tasks) {
        for (size_t i = 0; i < nb_threads; ++i) {
            add_task(tasks[i], i);
        }
    }
    /*wait for a task to be finished*/
    void wait_task(size_t id){
        //we ensure the thread started the task
        while (!started[id]) {
            usleep(100);
        }
        //we lock the do mutex
        mtx_do[id].lock();
        //we lock the finished mutex
        mtx_finished[id].lock();
        //we set the started bool to false so the thread will stop waiting for it to be false and will be waiting for the do mutex to be unlocked
        started[id] = false;

    }
    /*wait for all tasks to be finished*/
    void wait_all_tasks() {
        std::vector<bool> joined(nb_threads);
        for (size_t i = 0; i < nb_threads; ++i) {
            joined[i] = false;
        }
        size_t nb_joined = 0;
        while (nb_joined < nb_threads) {
            //std::cout << nb_joined << " tasks joined" << std::endl;
            for (size_t i = 0; i < nb_threads; ++i) {
                if(!joined[i]){
                    if (started[i]) {
                        //std::cout << "joining task : " << i << std::endl;
                        wait_task(i);
                        joined[i] = true;
                        ++nb_joined;
                        //std::cout << "joined task : " << i << std::endl;
                    }
                }
            }
            if (nb_joined < nb_threads) {
                usleep(100);
            }            
        }
        //std::cout << "all tasks joined" << std::endl;
    }
    void clear_tasks() {
        for (size_t i = 0; i < nb_threads; ++i) {
            // kill the thread
            kiled[i] = true;
            // create a null task
            std::function<void()> task = []() {};
            // add the null task to the thread
            add_task(task, i);
            // join the thread
            threads[i].join();
        }
    
    }
};