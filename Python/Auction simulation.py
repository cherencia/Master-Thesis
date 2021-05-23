#Auction simulation 
import numpy as np
import math as mth
from numpy.core.defchararray import index
from numpy.core.fromnumeric import shape
from numpy.core.records import record
from numpy.lib.function_base import append


class Auctioneer:# Class that draw the different offers 
    
    def __init__(self, k, T, J, M, tau, alpha, beta):
        
        self.k = k
        self.N = T
        self.t = 0
        self.J = J
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.values = []
        self.CTR_naive = np.zeros(shape=self.k)
        self.CTR_shrink = np.zeros(shape=self.k)
        self.CTR_bayes = np.zeros(shape=self.k)
        self.sigmas = np.zeros(shape=self.k)
        self.alphas = []
        self.betas = []
        self.successes_naive = np.zeros(shape=self.k)
        self.successes_shrink = np.zeros(shape=self.k)
        self.successes_bayes = np.zeros(shape=self.k)
        self.failures_naive = np.zeros(shape=self.k)
        self.failures_shrink = np.zeros(shape=self.k)
        self.failures_bayes = np.zeros(shape=self.k)
        self.regretNaive = 0
        self.regretShrink = 0
        self.regretBayes = 0
        self.w1 = np.zeros(shape=self.k)
        self.w2 = np.zeros(shape=self.k)
        self.w3 = np.zeros(shape=self.k)
    
    def draw_values(self): # Draw the theta values from a beta distribution
        for i in range(self.k):
            self.values.append(np.random.beta(self.alpha, self.beta))
        for i in range(self.k):
            self.alphas.append(self.alpha)
            self.betas.append(self.beta)
        self.values.sort(reverse=True)
    
    def SimulateClicksNaive(self): # display ads according to the schedule w
        successes_period = 0
        loss = 0
        for k in range(self.k):
            successes = np.sum(np.random.binomial(1, self.values[k], int(self.M*self.w1[k])))
            # successes_period += successes
            self.successes_naive[k] = self.successes_naive[k] + successes
            self.alphas[k] = self.alphas[k] + successes
            failures = int(self.M*self.w1[k])-successes
            self.failures_naive[k] = self.failures_naive[k] + failures
            self.betas[k] = self.betas[k] + failures
            self.CTR_naive[k] = self.successes_naive[k]/(self.failures_naive[k]+ self.successes_naive[k])
            sigma = mth.sqrt((self.CTR_naive[k]*(1-self.CTR_naive[k])/(self.failures_naive[k]+ self.successes_naive[k])))
            self.sigmas[k] = sigma
        for k in range(self.k):
            loss += self.w1[k]*self.M*self.values[k]

        regret = self.M * max(self.values) - loss
        self.regretNaive = self.regretNaive + regret
    
    def SimulateClicksShrink(self): # display ads according to the schedule w
        # successes_period = 0
        loss = 0
        for k in range(self.k):
            successes = np.sum(np.random.binomial(1, self.values[k], int(self.M*self.w2[k])))
            # successes_period += successes
            self.successes_shrink[k] = self.successes_shrink[k] + successes
            self.alphas[k] = self.alphas[k] + successes
            failures = int(self.M*self.w2[k])-successes
            self.failures_shrink[k] = self.failures_shrink[k] + failures
            self.betas[k] = self.betas[k] + failures
            self.CTR_shrink[k] = self.successes_shrink[k]/(self.failures_shrink[k]+ self.successes_shrink[k])
            sigma = mth.sqrt((self.CTR_shrink[k]*(1-self.CTR_shrink[k])/(self.failures_shrink[k]+ self.successes_shrink[k])))
            self.sigmas[k] = sigma
        for k in range(self.k):
            loss += self.w2[k]*self.M*self.values[k]

        regret = self.M * max(self.values) - loss
        self.regretShrink = self.regretNaive + regret
    
    def SimulateClicksBayes(self): # display ads according to the schedule w
        # successes_period = 0
        loss = 0
        for k in range(self.k):
            successes = np.sum(np.random.binomial(1, self.values[k], int(self.M*self.w3[k])))
            # successes_period += successes
            self.successes_bayes[k] = self.successes_bayes[k] + successes
            self.alphas[k] = self.alphas[k] + successes
            failures = int(self.M*self.w3[k])-successes
            self.failures_bayes[k] = self.failures_bayes[k] + failures
            self.betas[k] = self.betas[k] + failures
            self.CTR_bayes[k] = self.successes_bayes[k]/(self.failures_bayes[k]+ self.successes_bayes[k])
            sigma = mth.sqrt((self.CTR_bayes[k]*(1-self.CTR_bayes[k])/(self.failures_bayes[k]+ self.successes_bayes[k])))
            self.sigmas[k] = sigma
        for k in range(self.k):
            loss += self.w3[k]*self.M*self.values[k]

        regret = self.M * max(self.values) - loss
        self.regretBayes = self.regretNaive + regret
    
    def NaiveSelection(self,): # generate a schedule according to naive sel
        
        if self.t < self.tau:
            self.w1 = np.zeros(shape=self.k)
            for k in range(self.k):
                self.w1[k] = 1/self.k
        elif self.t == self.tau:
            self.w1 = np.zeros(shape=self.k)
            winner = np.argmax(self.CTR_naive)
            self.w1[winner] = 1
        else: 
            pass


    def ShrinkageSelection(self, c):# generate a schedule accordint to shirnk sel

        
        p_hat =[]
        for i in range(self.k):
            shrinked_offer = self.CTR_shrink[i]-c*self.sigmas[i]
            p_hat.append(shrinked_offer)
        winner = p_hat.index(max(p_hat))
        if self.t < self.tau:
            self.w2 = np.zeros(shape=self.k)
            for k in range(self.k):
                self.w2[k] = 1/self.k
        elif self.t == self.tau:
            self.w2 = np.zeros(shape=self.k)
            self.w2[winner] = 1
        else: 
            pass
        
    # our modified version of the bandits
    def BayesianBandits(self):

        I = np.zeros(shape=(self.k, self.J))
        self.w3 = np.zeros(shape=self.k)
        for j in range(self.J):
            thetas = []
            for k in range(self.k):
                thetas.append(np.random.beta(self.alphas[k], self.betas[k]))
            winner = thetas.index(max(thetas))
            I[winner, j] = 1
            
        self.w3 = I.sum(axis=1)/self.J
    
    def NextPeriod(self):
        self.t += 1
    
# unit test
# intitialize the auction
test = Auctioneer(3, 10, 100, 10, 3, 0.98, 5.3186)
test.draw_values()
test.BayesianBandits()
test.NaiveSelection()
test.ShrinkageSelection(0.5)
test.SimulateClicksBayes()
test.SimulateClicksNaive()
test.NextPeriod()
# check values and offers
test.values

# function that simulates many auctions and records the results

# Function that returns a matrix with the selectivity for each policy
def SimulationComparison(k, T, J, M, tau, alpha, beta, draws):
    
    c_values = np.linspace(0,1.5,12)
    # result matrix
    selectivity_matrix_naive = np.zeros(shape=(draws))
    selectivity_matrix_shrink = np.zeros(shape=(draws,len(c_values)))
    for i in range(draws):# we perform 1000 simulated scenarios for each c
        print(i)
        for c in range(len(c_values)):
            auction = Auctioneer(k, T, J, M, tau, alpha, beta)
            auction.draw_values()
            for period in range(T):
                auction.BayesianBandits()
                auction.NaiveSelection()
                auction.ShrinkageSelection(c)
                auction.SimulateClicksBayes()
                auction.SimulateClicksNaive()
                auction.SimulateClicksShrink()
                auction.NextPeriod()
            if np.argmax(auction.w1) == 0:
                selectivity_matrix_naive[i] = 1
            else:
                selectivity_matrix_naive[i] = 0
            if np.argmax(auction.w2) == 0:
                selectivity_matrix_shrink[i, c] = 1
            else:
                selectivity_matrix_shrink[i, c] = 0
    return selectivity_matrix_naive, selectivity_matrix_shrink

wasabi1, wasabi2 = SimulationComparison(3, 4, 100, 4, 3, 0.98, 5.3186, 10)

def RunExperimentsSelectivity(T, J, tau, alpha, beta, draws):

    # running auctions with different parameters
    # and saving them to different .csv files
    for M in (100,1000):
        for k in range(3,9):
            data1, data2 = SimulationComparison(k, T, J, M, tau, alpha, beta, draws)
            np.savetxt('results_naive_' + str(M) + '_' + str(k), data1, delimiter=',')
            np.savetxt('results_shrink' + str(M) + '_' + str(k), data2, delimiter=',')

            

test1 = RunExperimentsSelectivity(4, 100, 3, 0.98, 5.3186, 10000)



# regrets for the shrink selection
def shrinkRegrets(k, T, J, M, tau, alpha, beta, draws):
    # result matrix
    c_values = np.linspace(0,1.5,12)
    regret_matrix = np.zeros(shape=(draws, len(c_values)))
    for i in range(draws):# we perform 1000 simulated scenarios for each c
        print(i)
        for c in range(len(c_values)):
            auction = Auctioneer(k, T, J, M, tau, alpha, beta)
            auction.draw_values()
            for period in range(T):
                auction.ShrinkageSelection(c)
                auction.SimulateClicksShrink()
                auction.NextPeriod()
            regret_matrix[i, c] = auction.regretShrink
    return regret_matrix

# unit test for shrink
test_shrink = shrinkRegrets(16, 10, 1000, 1000, 1, 1, 5, 100)

def RunExperimentsShrinkRegrets(T, J, tau, alpha, beta, draws):

    # running auctions with different parameters
    # and saving them to different .csv files
    for M in (100,1000):
        for k in range(3,9):
            data1 = shrinkRegrets(k, T, J, M, tau, alpha, beta, draws)
            np.savetxt('results_regret_shrink' + str(M) + '_' + str(k), data1, delimiter=',')

shrink_regrets = RunExperimentsShrinkRegrets(10, 1000, 1, 0.98, 5.3, 1000)

# function that compare regrets
def compareRegrets(k, T, J, M, tau, alpha, beta, draws):
    # result matrix
    regret_matrix = np.zeros(shape=(draws, 2))
    
    for i in range(draws):# we perform 1000 simulated scenarios for each c
        print(i)
        auction = Auctioneer(k, T, J, M, tau, alpha, beta)
        auction.draw_values()
        for period in range(T):
            auction.BayesianBandits()
            auction.NaiveSelection()
            auction.SimulateClicksBayes()
            auction.SimulateClicksNaive()
            auction.NextPeriod()
        regret_matrix[i, 0] = auction.regretNaive
        regret_matrix[i, 1] = auction.regretBayes
    
    return regret_matrix

test2 = compareRegrets(3, 50, 1000, 100, 1, 1, 1, 10)

def RunExperimentsRegret(T, J, tau, alpha, beta, draws):

    # running auctions with different parameters
    # and saving them to different .csv files
    for M in (100,1000):
        for k in range(3,9):
            data = compareRegrets(k, T, J, M, tau, alpha, beta, draws)
            np.savetxt('results_regrets_' + str(M) + '_' + str(k), data, delimiter=',')

experiments_regret = RunExperimentsRegret(10, 1000, 1, 0.98, 5.3, 1000)

# function that records the wieghts from the bayesian bandits

def recordWeights(k, T, J, M, tau, alpha, beta, draws):
    # we are going to create a list of list of weight arraws
    list_of_matrixes = []
    for i in range(draws):# we perform 1000 
        print(i)
        matrix_weights =np.zeros(shape=(T, k))
        auction = Auctioneer(k, T, J, M, tau, alpha, beta)
        auction.draw_values()
        for period in range(T):
            auction.BayesianBandits()
            auction.SimulateClicksBayes()
            matrix_weights[auction.t] = auction.w3 
            auction.NextPeriod()
        list_of_matrixes.append(matrix_weights)
        
    
    return list_of_matrixes

matrixes_w_bayes = recordWeights(3, 10, 1000, 100, 3, 0.98, 5.3, 1000)

def calculateMeanWeights(list_of_matrix, T, k, draws):
    final_array = np.zeros(shape=(T, k))
    for matrix in list_of_matrix:
        for row in  range(matrix.shape[0]):
            final_array[row] += matrix[row]
    return final_array/draws


final_matrix_w_bayes = calculateMeanWeights(matrixes_w_bayes, 10, 3, 1000)
np.savetxt('final_matrix_bayes' ,final_matrix_w_bayes, delimiter=',')


