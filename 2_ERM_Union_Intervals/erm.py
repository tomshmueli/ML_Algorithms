#################################
# Your name: Tom Shmueli - 315363473
#################################
import math

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.random.uniform(0, 1, m)
        X.sort()
        Y = np.array([np.random.choice([0, 1], p=self.probability_y_given_x(x)) for x in X])
        return np.column_stack((X, Y))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        average_errors = []
        for n in range(m_first, m_last + 1, step):
            ep_es_for_experiment = [self.calculate_ep_es(n, k) for t in range(T)]
            average_ep_es = [sum(error) / T for error in zip(*ep_es_for_experiment)]
            average_errors.append(average_ep_es)
        np_avg_errors = np.asarray(average_errors)
        ns = np.arange(m_first, m_last + 1, step)

        plt.title("Experiment m range erm")
        plt.xlabel("n")
        plt.plot(ns, np_avg_errors[:, 0], marker='o', label="true error")
        plt.plot(ns, np_avg_errors[:, 1], marker='o', label="empirical error")
        plt.legend()
        plt.show()

        return np_avg_errors

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        errors_lst = []
        sample_m = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            ep_es_for_curr_k = self.calculate_ep_es(m, k, isSample=False, sample=sample_m)
            errors_lst.append(ep_es_for_curr_k)
        np_errors = np.asarray(errors_lst)
        best_k = np.argmin(np_errors[:, 0]) * step + k_first
        ks = np.arange(k_first, k_last + 1, step)

        plt.title("Experiment k range erm")
        plt.xlabel("k")
        plt.plot(ks, np_errors[:, 0], marker='o', label="true error")
        plt.plot(ks, np_errors[:, 1], marker='o', label="empirical error")
        plt.legend()
        plt.show()

        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        errors_lst = []
        penalty_lst = []
        srm_sum_lst = []
        i = 0
        sample_m = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            vc_dim = k + 1
            penalty = 2 * math.sqrt((vc_dim + math.log(2 / 0.1)) / m)
            ep_es_for_curr_k = self.calculate_ep_es(m, k, isSample=False, sample=sample_m)
            errors_lst.append(ep_es_for_curr_k)
            penalty_lst.append(penalty)
            srm_sum_lst.append(errors_lst[i][1] + penalty)
            i += 1

        combined_data = np.column_stack((errors_lst, penalty_lst, srm_sum_lst))
        best_k = np.argmin(combined_data[:, 0]) * step + k_first
        ks = np.arange(k_first, k_last + 1, step)

        plt.title("Experiment k range SRM")
        plt.xlabel("k")
        plt.plot(ks, combined_data[:, 0], marker='o', label="true error")
        plt.plot(ks, combined_data[:, 1], marker='o', label="empirical error")
        plt.plot(ks, combined_data[:, 2], marker='o', label="penalty")
        plt.plot(ks, combined_data[:, 3], marker='o', label="es+penalty")
        plt.legend()
        plt.show()

        return best_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # split data set to training and holdout--> 80%-20%
        sample_m = self.sample_from_D(m)
        np.random.shuffle(sample_m)
        split_index = int(0.8 * len(sample_m))
        train_data = sample_m[:split_index]
        holdout_data = sample_m[split_index:]
        train_data = train_data[train_data[:, 0].argsort()]
        holdout_data = holdout_data[holdout_data[:, 0].argsort()]

        erm_lst = []
        es_lst = []

        for k in range(1, 11):
            erm_and_es = intervals.find_best_interval(train_data[:, 0], train_data[:, 1], k)
            erm_lst.append(erm_and_es[0])

        for i in range(10):
            es = sum([self.zero_one_loss(erm_lst[i], x, y) for x, y in holdout_data]) / len(holdout_data)
            es_lst.append(es)

        best_k = es_lst.index(min(es_lst)) + 1
        return best_k

    #################################
    # Place for additional methods

    def probability_y_given_x(self, x):
        """From the data in the question
        Input: x E [0,1]
        Returns: Pr[y=1|X=x]
        """
        if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
            return [0.2, 0.8]
        return [0.9, 0.1]

    def interval_overlap(self, intervals1, intervals2):
        """Calculate the overlap between two intervals.
        Input:
        intervals1: list of tuples
        intervals2: list of tuples
        Returns:
        overlap: length of the overlap between the intervals
        """
        overlap = 0
        p1 = 0
        p2 = 0
        while p1 < len(intervals1) and p2 < len(intervals2):
            start = max(intervals1[p1][0], intervals2[p2][0])
            end = min(intervals1[p1][1], intervals2[p2][1])
            if start < end:
                overlap += (end - start)
            if intervals1[p1][1] == intervals2[p2][1]:
                p1 += 1
                p2 += 1
            elif intervals1[p1][1] < intervals2[p2][1]:
                p1 += 1
            else:
                p2 += 1

        return overlap

    def calculate_ep_es(self, n, k, isSample=True, sample=None):
        """Calculate the true error and the empirical error given from a random sample n
        Input:
        n[int]: size of data sample
        k[int]: number of intervals
        isSample[bool] - flag to create new sample set size n
        sample[np.ndarray of shape (m,2)] - a sample given from experiment
        Returns:
        [tupple] - (ep,es)
        """
        if isSample:
            sample = self.sample_from_D(n)
        # calculate empirical error es:
        best_intervals, best_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
        es = best_error / n  # average of errors

        y1_high_pr = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        y1_low_pr = [(0.2, 0.4), (0.6, 0.8)]
        len_cur_and_high_pr = self.interval_overlap(best_intervals,
                                                    y1_high_pr)
        len_cur_and_low_pr = self.interval_overlap(best_intervals,
                                                   y1_low_pr)
        complement_cur_and_high_pr = 0.6 - len_cur_and_high_pr
        complement_cur_and_low_pr = 0.4 - len_cur_and_low_pr
        ep = 0.8 * complement_cur_and_high_pr + 0.1 * complement_cur_and_low_pr + \
             0.2 * len_cur_and_high_pr + 0.9 * len_cur_and_low_pr

        return ep, es

    def zero_one_loss(self, list_intervals, x, y):
        """Calculate zero_one_loss
        Input: list_intervals - a list of tuples, every tuple is an interval.
            x[float] - from 0 to 1.
            y[int] - an integer, 0 or 1

        Returns: 0 if h(x)=y, else 1
        """
        x_in_interval = False
        for interval in list_intervals:
            if interval[0] <= x <= interval[1]:
                x_in_interval = True

        if (x_in_interval and y == 1) or (not x_in_interval and y == 0):
            return 0
        return 1



if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
