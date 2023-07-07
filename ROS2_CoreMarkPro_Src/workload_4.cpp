#include "rclcpp/rclcpp.hpp"
#include <sched.h>
#include <unistd.h>

#include <fstream> // Include the <fstream> header
#include <iostream>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/wait.h>

extern "C" int radix_main(int argc, char *argv[]);

class Load4: public rclcpp::Node
{
public:
    Load4(): Node("radix_test")
    {
         RCLCPP_INFO(this->get_logger(), "START ** ");

         timer_ = this->create_wall_timer(std::chrono::microseconds(5000), std::bind(&Load4::timerCallback, this));
         outputFile_.open("radix_PMC.csv");

         if (!outputFile_.is_open()) {
             RCLCPP_ERROR(this->get_logger(), "Failed to open output file");
         }
    }
    ~Load4() {
    // Close the output file
    if (outputFile_.is_open()) {
      outputFile_.close();
    }
    }    
     
private:

    void timerCallback()
    {
        counter_++;
        float period = 5000.;

        RCLCPP_INFO(this->get_logger(), "Hello, round: %d", counter_);

        int argc=4;
        char *argv[] = { "-v0", "-i1", "-c1", "-w1" };

        ////////////////////////////////////////////////////////////////////////////////
        //                              MEASURE L1 CACHE                             ///
        ////////////////////////////////////////////////////////////////////////////////

        // Create perf event attributes for L1 dcache load misses
        struct perf_event_attr l1d_attr{};
        l1d_attr.type = PERF_TYPE_HW_CACHE;
        l1d_attr.size = sizeof(l1d_attr);
        l1d_attr.config = (PERF_COUNT_HW_CACHE_L1D |
                          (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        l1d_attr.disabled = 1;
        l1d_attr.exclude_kernel = 1;
        l1d_attr.exclude_hv = 1;

        // Create perf event attributes for L1 dcache load
        struct perf_event_attr l1d_load_attr{};
        l1d_load_attr.type = PERF_TYPE_HW_CACHE;
        l1d_load_attr.size = sizeof(l1d_load_attr);
        l1d_load_attr.config = (PERF_COUNT_HW_CACHE_L1D |
                          (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                          (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
        l1d_load_attr.disabled = 1;
        l1d_load_attr.exclude_kernel = 1;
        l1d_load_attr.exclude_hv = 1;

        //  TODO: DEBUG
        // // Create perf event attributes for L1 dcache stores
        // struct perf_event_attr l1_dcache_stores_attr{};
        // l1_dcache_stores_attr.type = PERF_TYPE_HW_CACHE;
        // l1_dcache_stores_attr.size = sizeof(struct perf_event_attr);
        // l1_dcache_stores_attr.config = PERF_COUNT_HW_CACHE_L1D |
        //                                 (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        //                                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        // l1_dcache_stores_attr.disabled = 1;
        // l1_dcache_stores_attr.exclude_kernel = 1;
        // l1_dcache_stores_attr.exclude_hv = 1;


        // Create perf event attributes for L1 icache load misses
        struct perf_event_attr l1_icache_attr{};
        l1_icache_attr.type = PERF_TYPE_HW_CACHE;
        l1_icache_attr.size = sizeof(l1_icache_attr);
        l1_icache_attr.config = PERF_COUNT_HW_CACHE_L1I |
                                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        l1_icache_attr.disabled = 1;
        l1_icache_attr.exclude_kernel = 1;
        l1_icache_attr.exclude_hv = 1;


        // // Create perf event attributes for L1 icache load
        // struct perf_event_attr l1_icache_load_attr{};
        // l1_icache_load_attr.type = PERF_TYPE_HW_CACHE;
        // l1_icache_load_attr.size = sizeof(l1_icache_load_attr);
        // l1_icache_load_attr.config = PERF_COUNT_HW_CACHE_L1I |
        //                         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        //                         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
        // l1_icache_load_attr.disabled = 1;
        // l1_icache_load_attr.exclude_kernel = 1;
        // l1_icache_load_attr.exclude_hv = 1;



        ////////////////////////////////////////////////////////////////////////////////
        //                              MEASURE LLC                                  ///
        //  Attention: cannot directly measure LLC load and LLC load misses together ///
        //  can not measure LLC-store-misses LLC-stores                              ///
        ////////////////////////////////////////////////////////////////////////////////


        // Create perf event attributes for LLC load misses
        struct perf_event_attr llc_attr{};
        llc_attr.type = PERF_TYPE_HW_CACHE;
        llc_attr.size = sizeof(llc_attr);
        llc_attr.config = (PERF_COUNT_HW_CACHE_LL |
                          (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                          (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
        llc_attr.disabled = 1;
        llc_attr.exclude_kernel = 1;
        llc_attr.exclude_hv = 1;

         
        // // Create perf event attributes for LLC loads
        // struct perf_event_attr llc_loads_attr{};
        // llc_loads_attr.type = PERF_TYPE_HW_CACHE;
        // llc_loads_attr.size = sizeof(llc_loads_attr);
        // llc_loads_attr.config = PERF_COUNT_HW_CACHE_LL |
        //                         (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        //                         (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
        // llc_loads_attr.disabled = 1;
        // llc_loads_attr.exclude_kernel = 1;
        // llc_loads_attr.exclude_hv = 1;        
        

        // // Create perf event attributes for LLC LLC-store-misses
        // struct perf_event_attr llc_store_misses_attr{};
        // llc_store_misses_attr.type = PERF_TYPE_HW_CACHE;
        // llc_store_misses_attr.size = sizeof(struct perf_event_attr);
        // llc_store_misses_attr.config = PERF_COUNT_HW_CACHE_LL |
        //                                 (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        //                                 (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        // llc_store_misses_attr.disabled = 1;
        // llc_store_misses_attr.exclude_kernel = 1;
        // llc_store_misses_attr.exclude_hv = 1;



        ////////////////////////////////////////////////////////////////////////////////
        //                              MEASURE Hardware event                       ///
        //     branch-instructions, branch-misses, cpu-cycles, instructions          ///
        //  ref-cycles, bus-cycles, (cache-misses and cache-references unavalabel)   ///
        ////////////////////////////////////////////////////////////////////////////////

        // Create perf event attributes for branch misses
        struct perf_event_attr branch_misses_attr{};
        branch_misses_attr.type = PERF_TYPE_HARDWARE;
        branch_misses_attr.size = sizeof(branch_misses_attr);
        branch_misses_attr.config = PERF_COUNT_HW_BRANCH_MISSES;
        branch_misses_attr.disabled = 1;
        branch_misses_attr.exclude_kernel = 1;
        branch_misses_attr.exclude_hv = 1;

        struct perf_event_attr branch_instructions_attr{};
        branch_instructions_attr.type = PERF_TYPE_HARDWARE;
        branch_instructions_attr.size = sizeof(branch_instructions_attr);
        branch_instructions_attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
        branch_instructions_attr.disabled = 1;
        branch_instructions_attr.exclude_kernel = 1;
        branch_instructions_attr.exclude_hv = 1;

        // struct perf_event_attr cache_misses_attr{};
        // cache_misses_attr.type = PERF_TYPE_HARDWARE;
        // cache_misses_attr.size = sizeof(cache_misses_attr);
        // cache_misses_attr.config = PERF_COUNT_HW_CACHE_MISSES;
        // cache_misses_attr.disabled = 1;
        // cache_misses_attr.exclude_kernel = 1;
        // cache_misses_attr.exclude_hv = 1;

        // struct perf_event_attr cache_references_attr{};
        // cache_references_attr.type = PERF_TYPE_HARDWARE;
        // cache_references_attr.size = sizeof(cache_references_attr);
        // cache_references_attr.config = PERF_COUNT_HW_CACHE_REFERENCES;
        // cache_references_attr.disabled = 1;
        // cache_references_attr.exclude_kernel = 1;
        // cache_references_attr.exclude_hv = 1;

        struct perf_event_attr cpu_cycles_attr{};
        cpu_cycles_attr.type = PERF_TYPE_HARDWARE;
        cpu_cycles_attr.size = sizeof(struct perf_event_attr);
        cpu_cycles_attr.config = PERF_COUNT_HW_CPU_CYCLES;
        cpu_cycles_attr.disabled = 1;
        cpu_cycles_attr.exclude_kernel = 1;
        cpu_cycles_attr.exclude_hv = 1;

        struct perf_event_attr instructions_attr{};
        instructions_attr.type = PERF_TYPE_HARDWARE;
        instructions_attr.size = sizeof(struct perf_event_attr);
        instructions_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
        instructions_attr.disabled = 1;
        instructions_attr.exclude_kernel = 1;
        instructions_attr.exclude_hv = 1;

        struct perf_event_attr ref_cycles_attr{};
        ref_cycles_attr.type = PERF_TYPE_HARDWARE;
        ref_cycles_attr.size = sizeof(struct perf_event_attr);
        ref_cycles_attr.config = PERF_COUNT_HW_REF_CPU_CYCLES;
        ref_cycles_attr.disabled = 1;
        ref_cycles_attr.exclude_kernel = 1;
        ref_cycles_attr.exclude_hv = 1;

        struct perf_event_attr bus_cycles_attr{};
        bus_cycles_attr.type = PERF_TYPE_HARDWARE;
        bus_cycles_attr.size = sizeof(struct perf_event_attr);
        bus_cycles_attr.config = PERF_COUNT_HW_BUS_CYCLES;
        bus_cycles_attr.disabled = 1;
        bus_cycles_attr.exclude_kernel = 1;
        bus_cycles_attr.exclude_hv = 1;



        ////////////////////////////////////////////////////////////////////////////////
        //                              MEASURE Software event                       ///
        //                              cpu-clock,  task-clock                       ///
        //  (alignment-faults, bpf-output, cgroup-switches, context-switches,        ///
        //           cpu-migrations, page-faults, unavalabel)                        ///
        //                                                                           ///
        ////////////////////////////////////////////////////////////////////////////////

        struct perf_event_attr cpu_clock_attr{};
        cpu_clock_attr.type = PERF_TYPE_SOFTWARE;
        cpu_clock_attr.size = sizeof(struct perf_event_attr);
        cpu_clock_attr.config = PERF_COUNT_SW_CPU_CLOCK;
        cpu_clock_attr.disabled = 1;
        cpu_clock_attr.exclude_kernel = 1;
        cpu_clock_attr.exclude_hv = 1;


        struct perf_event_attr task_clock_attr{};
        task_clock_attr.type = PERF_TYPE_SOFTWARE;
        task_clock_attr.size = sizeof(struct perf_event_attr);
        task_clock_attr.config = PERF_COUNT_SW_TASK_CLOCK;
        task_clock_attr.disabled = 1;
        task_clock_attr.exclude_kernel = 1;
        task_clock_attr.exclude_hv = 1;



        ////////////////////////////////////////////////////////////////////////////////
        //                              MEASURE Kernel PMU event                       ///
        //                                   slots                                     ///
        // (mem-loads, topdown-bad-spec, topdown-be-bound, topdown-fe-bound, unavalable)  ///
        ////////////////////////////////////////////////////////////////////////////////

        struct perf_event_attr slots_attr{};
        slots_attr.type = PERF_TYPE_SOFTWARE;
        slots_attr.size = sizeof(struct perf_event_attr);
        slots_attr.config = PERF_COUNT_SW_CPU_CLOCK;
        slots_attr.disabled = 1;
        slots_attr.exclude_kernel = 1;
        slots_attr.exclude_hv = 1;


        // Open a perf event file descriptor

        ///////////////////////
        //      L1 CACHE     //
        ///////////////////////
        int l1d_fd = syscall(__NR_perf_event_open, &l1d_attr, 0, -1, -1, 0);
        int l1d_load_fd = syscall(__NR_perf_event_open, &l1d_load_attr, 0, -1, -1, 0);
        // int l1_dcache_stores_fd = syscall(__NR_perf_event_open, &l1_dcache_stores_attr, 0, -1, -1, 0);
        int l1_icache_fd = syscall(__NR_perf_event_open, &l1_icache_attr, 0, -1, -1, 0);
        // int l1_icache_load_fd = syscall(__NR_perf_event_open, &l1_icache_load_attr, 0, -1, -1, 0);


        ///////////////////////
        //         LLC       //
        ///////////////////////
        int llc_fd = syscall(__NR_perf_event_open, &llc_attr, 0, -1, -1, 0);
        // int llc_loads_fd = syscall(__NR_perf_event_open, &llc_loads_attr, 0, -1, -1, 0);
        // int llc_store_misses_fd = syscall(__NR_perf_event_open, &llc_store_misses_attr, 0, -1, -1, 0);


        ///////////////////////
        //  Hardware event   //
        ///////////////////////
        int branch_misses_fd = syscall(__NR_perf_event_open, &branch_misses_attr, 0, -1, -1, 0);
        int branch_instructions_fd = syscall(__NR_perf_event_open, &branch_instructions_attr, 0, -1, -1, 0);
        // int cache_misses_fd = syscall(__NR_perf_event_open, &cache_misses_attr, 0, -1, -1, 0);
        // int cache_references_fd = syscall(__NR_perf_event_open, &cache_references_attr, 0, -1, -1, 0);
        int cpu_cycles_fd = syscall(__NR_perf_event_open, &cpu_cycles_attr, 0, -1, -1, 0);
        int instructions_fd = syscall(__NR_perf_event_open, &instructions_attr, 0, -1, -1, 0);
        int ref_cycles_fd = syscall(__NR_perf_event_open, &ref_cycles_attr, 0, -1, -1, 0);
        int bus_cycles_fd = syscall(__NR_perf_event_open, &bus_cycles_attr, 0, -1, -1, 0);

        ///////////////////////
        //  Software event   //
        ///////////////////////
        int cpu_clock_fd = syscall(__NR_perf_event_open, &cpu_clock_attr, 0, -1, -1, 0);
        int task_clock_fd = syscall(__NR_perf_event_open, &task_clock_attr, 0, -1, -1, 0);   

        ///////////////////////
        // Kernel PMU event  //
        ///////////////////////
        int slots_fd = syscall(__NR_perf_event_open, &slots_attr, 0, -1, -1, 0);



           

        if (l1d_fd == -1 || l1d_load_fd == -1 || l1_icache_fd == -1 || llc_fd  == -1 ) { 
            // fd == -1 || llc_loads_fd
            std::cerr << "Failed to open perf event: cache-misses" << std::endl;
            return;
        }

        if (branch_instructions_fd == -1 || branch_misses_fd == -1) {
            std::cerr << "Failed to open perf event: branch instructions" << std::endl;
            return;
        }
        if (cpu_cycles_fd == -1 || instructions_fd == -1 || ref_cycles_fd == -1) {
            std::cerr << "Failed to open perf events" << std::endl;
            // Close previously opened file descriptors
            if (cpu_cycles_fd != -1)
                close(cpu_cycles_fd);
            if (instructions_fd != -1)
                close(instructions_fd);
            if (ref_cycles_fd != -1)
                close(ref_cycles_fd);
            return;
        }
        if (bus_cycles_fd == -1) {
            std::cerr << "Failed to open perf event: bus cycles" << std::endl;
            return;
        }
        // if (cache_references_fd == -1) {
        //     // cache_misses_fd == -1 || cache_references_fd == -1
        //     std::cerr << "Failed to open perf events" << std::endl;
        //     // Close previously opened file descriptors
        //     // if (cache_misses_fd != -1)
        //         // close(cache_misses_fd);
        //     if (cache_references_fd != -1)
        //         close(cache_references_fd);
        //     return;
        // }

        if (cpu_clock_fd == -1) {
            // || cpu_migrations_fd == -1

                std::cerr << "Failed to open perf events" << std::endl;
                // Close previously opened file descriptors
                if (cpu_clock_fd != -1)
                    close(cpu_clock_fd);
                // if (cpu_migrations_fd != -1)
                //     close(cpu_migrations_fd);
                return;
        }
        if (task_clock_fd == -1) {
            // page_faults_fd == -1 || 
            std::cerr << "Failed to open perf events" << std::endl;
            // Close previously opened file descriptors
            // if (page_faults_fd != -1)
            //     close(page_faults_fd);
            if (task_clock_fd != -1)
                close(task_clock_fd);
            return;
        }


        if (slots_fd == -1) {
            std::cerr << "Failed to open perf events" << std::endl;
            // Close previously opened file descriptors

            if (slots_fd != -1)
                close(slots_fd);

            return;
        }



        // Enable the perf event
        ///////////////////////
        //      L1 CACHE     //
        ///////////////////////
        ioctl(l1d_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(l1d_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(l1d_load_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(l1d_load_fd, PERF_EVENT_IOC_ENABLE, 0);
        // ioctl(l1_dcache_stores_fd, PERF_EVENT_IOC_RESET, 0);
        // ioctl(l1_dcache_stores_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(l1_icache_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(l1_icache_fd, PERF_EVENT_IOC_ENABLE, 0);
        // ioctl(l1_icache_load_fd, PERF_EVENT_IOC_RESET, 0);
        // ioctl(l1_icache_load_fd, PERF_EVENT_IOC_ENABLE, 0);

        ///////////////////////
        //         LLC       //
        ///////////////////////
        ioctl(llc_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(llc_fd, PERF_EVENT_IOC_ENABLE, 0);
        // ioctl(llc_loads_fd, PERF_EVENT_IOC_RESET, 0);
        // ioctl(llc_loads_fd, PERF_EVENT_IOC_ENABLE, 0);
        // ioctl(llc_store_misses_fd, PERF_EVENT_IOC_RESET, 0);
        // ioctl(llc_store_misses_fd, PERF_EVENT_IOC_ENABLE, 0);


        ///////////////////////
        //  Hardware event   //
        ///////////////////////
        ioctl(branch_misses_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(branch_misses_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(branch_instructions_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(branch_instructions_fd, PERF_EVENT_IOC_ENABLE, 0);
        // ioctl(cache_misses_fd, PERF_EVENT_IOC_RESET, 0);
        // ioctl(cache_misses_fd, PERF_EVENT_IOC_ENABLE, 0);
        // ioctl(cache_references_fd, PERF_EVENT_IOC_RESET, 0);
        // ioctl(cache_references_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(cpu_cycles_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(cpu_cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(instructions_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(instructions_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(ref_cycles_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(ref_cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(bus_cycles_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(bus_cycles_fd, PERF_EVENT_IOC_ENABLE, 0);

        ///////////////////////
        //  Software event   //
        ///////////////////////
        ioctl(cpu_clock_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(cpu_clock_fd, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(task_clock_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(task_clock_fd, PERF_EVENT_IOC_ENABLE, 0);


        ///////////////////////
        // Kernel PMU event  //
        ///////////////////////
        ioctl(slots_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(slots_fd, PERF_EVENT_IOC_ENABLE, 0);

        auto start = std::chrono::high_resolution_clock::now();
        auto start_timestamp = std::chrono::time_point_cast<std::chrono::microseconds>(start).time_since_epoch().count();

        /* first do abstraction layer specific initalizations */
        radix_main(argc, argv);

        auto end = std::chrono::high_resolution_clock::now();
        auto end_timestamp = std::chrono::time_point_cast<std::chrono::microseconds>(end).time_since_epoch().count();

        std::chrono::duration<double> duration = end - start;
        double executionTime = duration.count();

        // Disable the perf events and read the values

        ///////////////////////
        //      L1 CACHE     //
        ///////////////////////

        ioctl(l1d_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(l1d_load_fd, PERF_EVENT_IOC_DISABLE, 0);
        // ioctl(l1_dcache_stores_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(l1_icache_fd, PERF_EVENT_IOC_DISABLE, 0);
       

        ///////////////////////
        //         LLC       //
        ///////////////////////
        ioctl(llc_fd, PERF_EVENT_IOC_DISABLE, 0);
        // ioctl(llc_loads_fd, PERF_EVENT_IOC_DISABLE, 0);
        // ioctl(llc_store_misses_fd, PERF_EVENT_IOC_DISABLE, 0);


        // ioctl(dtlb_load_misses_fd, PERF_EVENT_IOC_DISABLE, 0);


        ///////////////////////
        //  Hardware event   //
        ///////////////////////
        ioctl(branch_misses_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(branch_instructions_fd, PERF_EVENT_IOC_DISABLE, 0);
        // ioctl(cache_misses_fd, PERF_EVENT_IOC_DISABLE, 0);
        // ioctl(cache_references_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(cpu_cycles_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(instructions_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(ref_cycles_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(bus_cycles_fd, PERF_EVENT_IOC_DISABLE, 0);


        ///////////////////////
        //  Software event   //
        ///////////////////////
        ioctl(cpu_clock_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(task_clock_fd, PERF_EVENT_IOC_DISABLE, 0);

        ///////////////////////
        // Kernel PMU event  //
        ///////////////////////
        ioctl(slots_fd, PERF_EVENT_IOC_DISABLE, 0);


        ///////////////////////
        //      L1 CACHE     //
        ///////////////////////
        uint64_t l1d_load_misses;
        uint64_t l1d_load;
        // uint64_t l1_dcache_stores;
        uint64_t l1_icache_load_misses;
        uint64_t l1_icache_load;

        ///////////////////////
        //      LLC          //
        ///////////////////////
        uint64_t llc_load_misses;
        // uint64_t llc_loads;
        // uint64_t llc_store_misses;



        ///////////////////////
        //  Hardware event   //
        ///////////////////////
        // uint64_t cache_misses;
        uint64_t branch_misses;
        uint64_t branch_instructions;
        uint64_t cache_misses, cache_references;
        uint64_t cpu_cycles, instructions, ref_cycles;
        uint64_t bus_cycles;

        ///////////////////////
        //  Software event   //
        ///////////////////////
        uint64_t cpu_clock;  
        uint64_t task_clock;

        ///////////////////////
        // Kernel PMU event  //
        ///////////////////////
        long long slots;



        ///////////////////////
        //      L1 CACHE     //
        ///////////////////////
        read(l1d_fd, &l1d_load_misses, sizeof(uint64_t));
        read(l1d_load_fd, &l1d_load, sizeof(uint64_t));
        // read(l1_dcache_stores_fd, &l1_dcache_stores, sizeof(uint64_t));
        read(l1_icache_fd, &l1_icache_load_misses, sizeof(uint64_t));
        // read(l1_icache_load_fd, &l1_icache_load, sizeof(uint64_t));

        ///////////////////////
        //      LLC          //
        ///////////////////////
        read(llc_fd, &llc_load_misses, sizeof(uint64_t));
        // read(llc_loads_fd, &llc_loads, sizeof(uint64_t));
        // read(llc_store_misses_fd, &llc_store_misses, sizeof(uint64_t));

        ///////////////////////
        //  Hardware event   //
        ///////////////////////
        read(branch_misses_fd, &branch_misses, sizeof(uint64_t));
        read(branch_instructions_fd, &branch_instructions, sizeof(uint64_t));
        // read(cache_misses_fd, &cache_misses, sizeof(uint64_t));
        // read(cache_references_fd, &cache_references, sizeof(uint64_t));
        read(cpu_cycles_fd, &cpu_cycles, sizeof(uint64_t));
        read(instructions_fd, &instructions, sizeof(uint64_t));
        read(ref_cycles_fd, &ref_cycles, sizeof(uint64_t));
        read(bus_cycles_fd, &bus_cycles, sizeof(uint64_t));


        ///////////////////////
        //  Software event   //
        ///////////////////////
        read(cpu_clock_fd, &cpu_clock, sizeof(uint64_t));
        read(task_clock_fd, &task_clock, sizeof(uint64_t));

        ///////////////////////
        // Kernel PMU event  //
        ///////////////////////
        read(slots_fd, &slots, sizeof(long long));



        // Close the perf event file descriptor
        ///////////////////////
        //      L1 CACHE     //
        ///////////////////////
        close(l1d_fd);
        close(l1d_load_fd);
        // close(l1_dcache_stores_fd);
        close(l1_icache_fd);
        // close(l1_icache_load_fd);


        ///////////////////////
        //      LLC          //
        ///////////////////////        
        close(llc_fd);
        // close(llc_loads_fd);
        // close(llc_store_misses_fd);


        ///////////////////////
        //  Hardware event   //
        ///////////////////////
        close(branch_misses_fd);
        close(branch_instructions_fd);
        // close(cache_misses_fd);
        // close(cache_references_fd);
        close(cpu_cycles_fd);
        close(instructions_fd);
        close(ref_cycles_fd);
        close(bus_cycles_fd);

        ///////////////////////
        //  Software event   //
        ///////////////////////
        close(cpu_clock_fd);
        close(task_clock_fd);

        ///////////////////////
        // Kernel PMU event  //
        ///////////////////////
        close(slots_fd);

        // auto clockFrequency = std::chrono::steady_clock::period::den /
        //                   std::chrono::steady_clock::period::num;


        std::cout << "Round: " << counter_ << std::endl;
        std::cout << "Task Period: " << period / 1000000. << std::endl;
        std::cout << "ExecutionTime: " << executionTime << std::endl;
        std::cout << "Task Uti: " <<  executionTime / (period / 1000000.) << std::endl;
        std::cout << "ExecutionStart time point: " << start_timestamp << std::endl;
        std::cout << "ExecutionStart end point: " << end_timestamp << std::endl;
        std::cout << "L1 dcache load misses: " << l1d_load_misses << std::endl;
        std::cout << "L1 dcache load: " << l1d_load << std::endl;
        // std::cout << "L1 dcache stores: " << l1_dcache_stores << std::endl;
        std::cout << "L1 icache load misses: " << l1_icache_load_misses << std::endl;
        // std::cout << "L1 icache load: " << l1_icache_load << std::endl;
        std::cout << "LLC load misses: " << llc_load_misses << std::endl;
        // std::cout << "LLC loads: " << llc_loads << std::endl;
        // std::cout << "LLC store misses: " << llc_store_misses << std::endl;
        std::cout << "Branch misses: " << branch_misses << std::endl;
        std::cout << "Branch instructions: " << branch_instructions << std::endl;
        // std::cout << "Cache misses: " << cache_misses << std::endl;
        // std::cout << "Cache references: " << cache_references << std::endl;
        std::cout << "CPU cycles: " << cpu_cycles << std::endl;
        std::cout << "Instructions: " << instructions << std::endl;
        std::cout << "Reference cycles: " << ref_cycles << std::endl;
        std::cout << "Bus cycles: " << bus_cycles << std::endl;
        std::cout << "CPU clock: " << cpu_clock << std::endl;
        std::cout << "Task clock: " << task_clock << std::endl;
        std::cout << "ExecutionTime: " << executionTime << std::endl;



        if (outputFile_.is_open()) {
            outputFile_ << "Round: " << counter_ << std::endl;
            outputFile_ << "Task Period: " << period / 1000000. << std::endl;
            outputFile_ << "ExecutionTime: " << executionTime << std::endl;
            outputFile_ << "Task Uti: " << executionTime / (period / 1000000.) << std::endl;
            outputFile_ << "ExecutionStart time point: " << start_timestamp << std::endl;
            outputFile_ << "ExecutionStart end point: " << end_timestamp << std::endl;
            outputFile_ << "L1 dcache load misses: " << l1d_load_misses << std::endl;
            outputFile_ << "L1 dcache load: " << l1d_load << std::endl;
            outputFile_ << "L1 icache load misses: " << l1_icache_load_misses << std::endl;
            outputFile_ << "LLC load misses: " << llc_load_misses << std::endl;
            outputFile_ << "Branch misses: " << branch_misses << std::endl;
            outputFile_ << "Branch instructions: " << branch_instructions << std::endl;
            outputFile_ << "CPU cycles: " << cpu_cycles << std::endl;
            outputFile_ << "Instructions: " << instructions << std::endl;
            outputFile_ << "Reference cycles: " << ref_cycles << std::endl;
            outputFile_ << "Bus cycles: " << bus_cycles << std::endl;
            outputFile_ << "CPU clock: " << cpu_clock << std::endl;
            outputFile_ << "Task clock: " << task_clock << std::endl;
            outputFile_ << "Slots: " << slots << std::endl;
            std::cout << "Metrics saved in " << "radix_PMC.csv" << std::endl;
        } else {
            std::cerr << "Failed to open output file" << std::endl;
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    int counter_;
    std::ofstream outputFile_;


};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Load4>();

  rclcpp::executors::SingleThreadedExecutor executor;
  
  // Set CPU affinity to core 0
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(1, &cpuset);  // Replace 0 with the desired core number

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