#include "rclcpp/rclcpp.hpp"

extern "C" int parser_125_main(int argc, char *argv[]);

class Load8: public rclcpp::Node
{
public:
    Load8(): Node("sha_test")
    {
         RCLCPP_INFO(this->get_logger(), "START ** ");

         timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&Load8::timerCallback, this));
    }
private:


    void timerCallback()
    {
        counter_++;
        RCLCPP_INFO(this->get_logger(), "Hello, round: %d", counter_);

        int argc=2;
        char *argv[] = { "-v0", "-i1" };
     
        /* first do abstraction layer specific initalizations */
        parser_125_main(argc, argv);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;


};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Load8>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}