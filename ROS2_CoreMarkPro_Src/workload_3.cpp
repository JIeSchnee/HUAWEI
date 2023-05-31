#include "rclcpp/rclcpp.hpp"
#include <sched.h>

extern "C" int loops_main(int argc, char *argv[]);

class Load3: public rclcpp::Node
{
public:
    Load3(): Node("loops_test")
    {
         RCLCPP_INFO(this->get_logger(), "START ** ");

         timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&Load3::timerCallback, this));
    }
private:


    void timerCallback()
    {
        counter_++;
        RCLCPP_INFO(this->get_logger(), "Hello, round: %d", counter_);

        int argc=2;
        char *argv[] = { "-v0", "-i1" };
     
        /* first do abstraction layer specific initalizations */
        loops_main(argc, argv);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;


};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Load3>();

  rclcpp::executors::SingleThreadedExecutor executor;
  
  // Set CPU affinity to core 0
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);  // Replace 0 with the desired core number

  pid_t pid = getpid();
  if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) == -1) {
    RCLCPP_ERROR(node->get_logger(), "Failed to set CPU affinity");
    return 1;
  }


  executor.add_node(node);

  executor.spin();
//   rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}