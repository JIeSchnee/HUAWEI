#include "rclcpp/rclcpp.hpp"
#include <sched.h>
#include <unistd.h>



extern "C" int linear_main(int argc, char *argv[]);

class Load2: public rclcpp::Node
{
public:
    Load2(): Node("linear_test")
    {
         RCLCPP_INFO(this->get_logger(), "START ** ");

         timer_ = this->create_wall_timer(std::chrono::microseconds(10000), std::bind(&Load2::timerCallback, this));
    }
private:


    void timerCallback()
    {
        counter_++;
        RCLCPP_INFO(this->get_logger(), "Hello, round: %d", counter_);

        int argc=2;
        char *argv[] = { "-v0", "-i1", "-c1", "-w1" };
     
        /* first do abstraction layer specific initalizations */
        linear_main(argc, argv);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;

};


// Function to set CPU core affinity for a thread
void setThreadAffinity(pthread_t thread, int cpuCore)
{

  std::cout << "cpuCore: " << cpuCore << std::endl;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpuCore, &cpuset);
  int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    // Handle error
  }
}

int getLowestCpuUsageCore()
{
  double loadavg[3];
  if (getloadavg(loadavg, 3) == -1) {
    // Handle error
  }

  int numCores = sysconf(_SC_NPROCESSORS_ONLN);
  double lowestUsage = loadavg[0];
  int lowestCore = 0;

  for (int core = 1; core < numCores; ++core) {
    if (loadavg[core] < lowestUsage) {
      lowestUsage = loadavg[core];
      lowestCore = core;
    }
  }
  std::cout << "lowestCore: " << lowestCore << std::endl;

  return lowestCore;
}

// In your ROS 2 node class
void allocateNodeToLowestCpuUsageCore()
{
  int lowestCore = getLowestCpuUsageCore();
  setThreadAffinity(pthread_self(), lowestCore);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Load2>();


  rclcpp::executors::SingleThreadedExecutor executor;
  
  // // Set CPU affinity to core 0
  // cpu_set_t cpuset;
  // CPU_ZERO(&cpuset);
  // CPU_SET(0, &cpuset);  // Replace 0 with the desired core number

  // pid_t pid = getpid();
  // if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) == -1) {
  //   RCLCPP_ERROR(node->get_logger(), "Failed to set CPU affinity");
  //   return 1;
  // }
  allocateNodeToLowestCpuUsageCore();

  executor.add_node(node);

  executor.spin();


//   rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}