import logging
import signal

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats
from sklearn.preprocessing import MinMaxScaler

from PythonCode.data_io import get_data, construct_dataframe
import seaborn as sns
from astropy.stats import bayesian_blocks
import ruptures as rpt
def enemy_workload_key(workload_name):
    # enemy_workloads = []
    sting = workload_name.split("_")
    if len(sting) > 3:
        new_name = "_".join(sting[-2:])
        temp = new_name.split(".")
        name = temp[0]
    else:
        new_name = "_".join(sting[-1:])
        temp = new_name.split(".")
        name = temp[0]
    # enemy_workloads.append(name)
    return name


def data_extraction_workload_based(data, monitor_items, analysis_workloads_name, datalist):
    for item in monitor_items:
        temp = {}
        if item == 'ExecutionStart time point':

            print("=====================================")
            print("Get {}".format(item))
            print("=====================================")
            mean_value_without_interference = data.query(f"workload_name == '{analysis_workloads_name}'")[
                item].diff().mean()
            std_value_without_interference = data.query(f"workload_name == '{analysis_workloads_name}'")[
                item].diff().std()

            for cmp_name in datalist:
                if cmp_name != analysis_workloads_name:
                    print("Workload name: ", cmp_name,
                          "-- diff mean and std value of {} without interference".format(item),
                          mean_value_without_interference, std_value_without_interference)
                    mean_value_with_interference = data.query(f"workload_name == '{cmp_name}'")[
                        item].diff().mean()
                    std_value_with_interference = data.query(f"workload_name == '{cmp_name}'")[
                        item].diff().std()

                    print("The diff mean and std value of {} with interference:".format(item),
                          mean_value_with_interference,
                          std_value_with_interference)

                    mean_diff = mean_value_with_interference - mean_value_without_interference
                    std_diff = std_value_with_interference - std_value_without_interference

                    percentage_mean = (mean_diff / mean_value_without_interference) * 100
                    percentage_std = (std_diff / std_value_without_interference) * 100

                    temp[enemy_workload_key(cmp_name)] = percentage_mean

                    print("Diff Mean value compared with the value of analysed workload {}:".format(mean_diff),
                          ". Ration",
                          percentage_mean)
                    print("Diff Std value compared with the value of analysed workload {}:".format(std_diff),
                          ". Ration",
                          percentage_std)
                    print("-------------------------------------")
        else:

            print("=====================================")
            print("Get {}".format(item))
            print("=====================================")

            mean_value_without_interference = data.query(f"workload_name == '{analysis_workloads_name}'")[
                item].mean()
            std_value_without_interference = data.query(f"workload_name == '{analysis_workloads_name}'")[
                item].std()

            for cmp_name in datalist:
                if cmp_name != analysis_workloads_name:
                    print("Workload name: ", cmp_name,
                          "-- mean and std value of {} without interference".format(item),
                          mean_value_without_interference, std_value_without_interference)
                    mean_value_with_interference = data.query(f"workload_name == '{cmp_name}'")[item].mean()
                    std_value_with_interference = data.query(f"workload_name == '{cmp_name}'")[item].std()

                    print("The mean and std value of {} with interference:".format(item),
                          mean_value_with_interference,
                          std_value_with_interference)

                    mean_diff = mean_value_with_interference - mean_value_without_interference
                    std_diff = std_value_with_interference - std_value_without_interference

                    percentage_mean = (mean_diff / mean_value_without_interference) * 100
                    percentage_std = (std_diff / std_value_without_interference) * 100

                    temp[enemy_workload_key(cmp_name)] = percentage_mean

                    print("Mean value compared with the value of analysed workload {}:".format(mean_diff), ". Ration",
                          percentage_mean)
                    print("Std value compared with the value of analysed workload {}:".format(std_diff), ". Ration",
                          percentage_std)
                    print("-------------------------------------")

        plt.bar(range(len(temp)), list(temp.values()), align='center')
        plt.xticks(range(len(temp)), list(temp.keys()))
        plt.ylabel("Diff Percentage")
        plt.xlabel("Enemy Workloads")
        plt.title("The difference of {}".format(item))
        plt.show()


def data_extraction_monitor_item_based(data, monitor_items, analysis_workloads_name, datalist):
    for workload_name in datalist:
        if workload_name != analysis_workloads_name:
            temp = {}
            if workload_name != analysis_workloads_name:
                print("=====================================")
                print(
                    "Get data from workload with interference from enemy {}".format(
                        enemy_workload_key(workload_name)))
                print("=====================================")
                for item in monitor_items:
                    print("Monitor item: ", item)

                    if item == 'ExecutionStart time point':

                        mean_value_without_interference = \
                            data.query(f"workload_name == '{analysis_workloads_name}'")[
                                item].diff().mean()
                        std_value_without_interference = \
                            data.query(f"workload_name == '{analysis_workloads_name}'")[
                                item].diff().std()

                        mean_value_with_interference = data.query(f"workload_name == '{workload_name}'")[
                            item].diff().mean()
                        std_value_with_interference = data.query(f"workload_name == '{workload_name}'")[
                            item].diff().std()

                        # use K-S test to check the distribution of data
                        ks_test_result = stats.ks_2samp(
                            data.query(f"workload_name == '{analysis_workloads_name}'")[item].diff().dropna(),
                            data.query(f"workload_name == '{workload_name}'")[item].diff().dropna())

                        mean_diff = mean_value_with_interference - mean_value_without_interference
                        std_diff = std_value_with_interference - std_value_without_interference

                        percentage_mean = (mean_diff / mean_value_without_interference) * 100
                        percentage_std = (std_diff / std_value_without_interference) * 100

                        temp[item] = percentage_mean

                        print("Workload name: ", workload_name,
                              "-- mean and std value of {} without interference".format(item),
                              mean_value_without_interference, std_value_without_interference)
                        print("The mean and std value of {} with interference:".format(item),
                              mean_value_with_interference,
                              std_value_with_interference)
                        print("Mean value compared with the value of analysed workload {}:".format(mean_diff),
                              ". Ration",
                              percentage_mean)
                        print("Std value compared with the value of analysed workload {}:".format(std_diff),
                              ". Ration",
                              percentage_std)
                        print("K-S test result: ", ks_test_result)
                        print("-------------------------------------")
                    else:

                        mean_value_without_interference = \
                            data.query(f"workload_name == '{analysis_workloads_name}'")[
                                item].mean()
                        std_value_without_interference = \
                            data.query(f"workload_name == '{analysis_workloads_name}'")[
                                item].std()

                        mean_value_with_interference = data.query(f"workload_name == '{workload_name}'")[
                            item].mean()
                        std_value_with_interference = data.query(f"workload_name == '{workload_name}'")[item].std()

                        # use K-S test to check the distribution of data
                        ks_test_result = stats.ks_2samp(
                            data.query(f"workload_name == '{analysis_workloads_name}'")[item].dropna(),
                            data.query(f"workload_name == '{workload_name}'")[item].dropna())

                        mean_diff = mean_value_with_interference - mean_value_without_interference
                        std_diff = std_value_with_interference - std_value_without_interference

                        percentage_mean = (mean_diff / mean_value_without_interference) * 100
                        percentage_std = (std_diff / std_value_without_interference) * 100

                        temp[item] = percentage_mean
                        print("Workload name: ", workload_name,
                              "-- mean and std value of {} without interference".format(item),
                              mean_value_without_interference, std_value_without_interference)
                        print("The mean and std value of {} with interference:".format(item),
                              mean_value_with_interference,
                              std_value_with_interference)
                        print("Mean value compared with the value of analysed workload {}:".format(mean_diff),
                              ". Ration",
                              percentage_mean)
                        print("Std value compared with the value of analysed workload {}:".format(std_diff),
                              ". Ration",
                              percentage_std)
                        print("K-S test result: ", ks_test_result)
                        print("-------------------------------------")

            # bar plot the temp dict
            plt.bar(range(len(temp)), list(temp.values()), align='center')
            plt.xticks(range(len(temp)), list(temp.keys()))
            plt.ylabel("Diff Percentage")
            plt.xlabel("Monitor items")
            plt.title("The difference of {}".format(workload_name))
            plt.show()


def preprocess(data, features, normaliser=None):
    # Normalise w.r.t. the CPU cycles.
    norm_features = ['L1 dcache load misses', 'L1 icache load misses',
                     'LLC load misses', 'Branch instructions', 'Branch misses', 'Bus cycles']
    data[norm_features] = data[norm_features].div(data["Instructions"], axis=0)

    # Normalise all features to 0-1.
    if normaliser is None:
        normaliser = MinMaxScaler()
        data[features] = normaliser.fit_transform(data[features])

        return data, features, normaliser
    else:
        data[features] = normaliser.transform(data[features])
        return data, features


def interference_analysis_and_visualisation(analysis_workloads_name, data, monitor_items, datalist):

    # for item in monitor_items:
    #     if item != "ExecutionStart time point":
    #         plt.plot(data.query(f"workload_name == '{analysis_workloads_name}'")[item], label=analysis_workloads_name)
    #         plt.legend()
    #         plt.title(item)
    #         plt.show()
    monitor_items = ['ExecutionTime', 'L1 dcache load misses', 'L1 icache load misses',
                     'LLC load misses', 'Branch instructions', 'Branch misses', 'Instructions', 'Bus cycles',
                     'CPU cycles']
    for item in monitor_items:
        for workload in datalist:
            if workload != analysis_workloads_name:
                d = data.query(f"workload_name == '{workload}'")[item].to_numpy()
                d_ref = data.query(f"workload_name == '{analysis_workloads_name}'")[item].to_numpy()

                # denoise
                # d = signal.medfilt(d, 3)
                # d_ref = signal.medfilt(d_ref, 3)

                temp = (d-d_ref)/d_ref
                plt.plot(np.arange(len(d)), temp, label=workload)
                algo = rpt.Window(model="l2", width=5)
                algo.fit(temp)
                result = algo.predict(n_bkps=5)

                rpt.display(temp, [], result)

                plt.title(f"{item} for '{workload}'")
                plt.show()



                # for i, kernel in enumerate(['linear', 'rbf', 'cosine']):
                #     algo = rpt.KernelCPD(kernel=kernel, min_size=5)
                #     algo.fit(temp)
                #     result = algo.predict(n_bkps=5)
                #     rpt.display(temp, [], result)
                #     # plt.plot(np.arange(len(d)), temp, label=workload)
                #     plt.title(f"{item} for '{workload}' with {kernel} kernel")
                #     plt.show()




def main():
    ##########################################
    # Load all data.
    ##########################################
    analysis_workloads_name = "nnest.csv"
    workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/core01/*')
    # workload_list = get_data(directory_path='/home/jiezou/ros2_ws/Data/Interf_analysis/*')
    original_data, features = construct_dataframe(workload_list)
    monitor_items = ['ExecutionStart time point', 'ExecutionTime', 'L1 dcache load misses', 'L1 icache load misses',
                     'LLC load misses', 'Branch instructions', 'Branch misses', 'Instructions', 'Bus cycles',
                     'CPU cycles']  # , 'Task clock', 'CPU clock'

    # data, features, normaliser = preprocess(original_data, features, normaliser=None)
    data = original_data.copy()
    interger_workloads = ['core', 'core_E', 'core_Enemy', 'sha', 'sha_E', 'sha_Enemy', 'zip', 'zip_E', 'zip_Enemy',
                          'parser', 'parser_E', 'parser_Enemy']
    floating_point_workloads = ['radix', 'linear', 'nnest', 'loops', 'radix_E', 'linear_E', 'nnest_E', 'loops_E',
                                'radix_Enemy', 'linear_Enemy', 'nnest_Enemy', 'loops_Enemy']
    analysis_workloads_period = 200000 / 1000000
    enemy_workloads_period = {}
    enemy_workloads_period['nnest_E'] = 300000 / 1000000
    enemy_workloads_period['nnest_Enemy'] = 1000000 / 1000000
    enemy_workloads_period['linear'] = 15000 / 1000000
    enemy_workloads_period['linear_E'] = 150000 / 1000000
    enemy_workloads_period['linear_Enemy'] = 70000 / 1000000
    enemy_workloads_period['loops'] = 300000 / 1000000
    enemy_workloads_period['loops_E'] = 1000000 / 1000000
    enemy_workloads_period['loops_Enemy'] = 2500000 / 1000000
    enemy_workloads_period['radix'] = 5000 / 1000000
    enemy_workloads_period['radix_E'] = 45000 / 1000000
    enemy_workloads_period['radix_Enemy'] = 25000 / 1000000
    enemy_workloads_period['zip'] = 250000 / 1000000
    enemy_workloads_period['zip_E'] = 500000 / 1000000
    enemy_workloads_period['zip_Enemy'] = 2500000 / 1000000
    enemy_workloads_period['core'] = 1500000 / 1000000
    enemy_workloads_period['core_E'] = 2500000 / 1000000
    enemy_workloads_period['core_Enemy'] = 4000000 / 1000000
    enemy_workloads_period['sha'] = 15000 / 1000000
    enemy_workloads_period['sha_E'] = 650000 / 1000000
    enemy_workloads_period['sha_Enemy'] = 300000 / 1000000
    enemy_workloads_period['parser'] = 55000 / 1000000
    enemy_workloads_period['parser_E'] = 1000000 / 1000000
    enemy_workloads_period['parser_Enemy'] = 400000 / 1000000

    # get the key of enemy enemy_workloads_period
    enemy_workloads_period_key = list(enemy_workloads_period.keys())

    ##########################################
    # Information.
    ##########################################
    logger.info('The length of analysed data: {}'.format(len(data)))
    logger.info("Get collected interference data: {}".format(data['workload_name'].unique()))
    logger.info('the period of analysis workloads is: {}'.format(analysis_workloads_period))
    logger.info('the period of enemy workloads is: {}'.format(enemy_workloads_period))
    datalist = data['workload_name'].unique()

    ##########################################
    # Interference analysis and visualisation.
    ##########################################
    interference_analysis_and_visualisation(analysis_workloads_name, data, monitor_items, datalist)







    ##########################################
    # Data processing and Analysis
    ##########################################
    # data_extraction_workload_based(data, monitor_items, analysis_workloads_name, datalist)
    # data_extraction_monitor_item_based(data, monitor_items, analysis_workloads_name, datalist)


  # for cmp_name in datalist:
    #     data.query(f"workload_name == '{cmp_name}'")[monitor_items].hist()
    #     plt.suptitle(cmp_name)
    #     plt.show()

    # print(data.loc[:, data.columns != 'workload_name'][monitor_items].std())

if __name__ == "__main__":
    logger = logging.getLogger()
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)
    main()
