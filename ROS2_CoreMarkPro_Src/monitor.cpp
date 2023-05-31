#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/wait.h>

class Monitor: public rclcpp::Node
{
public:
    Monitor(): Node("monitor")
    {
         RCLCPP_INFO(this->get_logger(), "START ** ");

         timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&Monitor::timerCallback, this));
    }
private:


    void timerCallback()
    {
         // Create perf event attributes for cache-misses
        struct perf_event_attr attr{};
        attr.type = PERF_TYPE_HW_CACHE;
        attr.size = sizeof(struct perf_event_attr);
        attr.config = PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;

        pid_t pid = 418739;

        // Open a perf event file descriptor for the process with PID
        int fd = syscall(__NR_perf_event_open, &attr, pid, -1, -1, 0);
        if (fd == -1) {
            std::cerr << "Failed to open perf event: cache-misses" << std::endl;
            return;
        }

        // Enable the perf event
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

        // Wait for the process to finish execution
        int status;
        waitpid(pid, &status, 0);

        // Disable the perf event and read the value
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        uint64_t cache_misses;
        read(fd, &cache_misses, sizeof(uint64_t));

        // Close the perf event file descriptor
        close(fd);

        std::cout << "Cache misses: " << cache_misses << std::endl;
            
            
        
        counter_++;
        RCLCPP_INFO(this->get_logger(), "Hello, round: %d", counter_);


    }

    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;


};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Monitor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}