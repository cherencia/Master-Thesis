#Auction simulation 
import numpy as np
import math as mth
from numpy.core.defchararray import index
from numpy.lib.function_base import append


class Auctioneer:# Class that draw the different offers and record the results
    
    def __init__(self, k, N, alpha, beta):
        
        self.k = k
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.values = []
        self.offers = []
        self.sigmas = []
        self.BayesianProbabilites = []
        self.impressions_first = 0
        self.bayes_arm_selected1 = []
        self.bayes_arm_selected2 = []
        self.alphas = []
        self.betas = []
        self.succeses_first_ad = 0
        
    
    def draw_values(self): # Draw the theta values from a beta distribution
        for i in range(self.k):
            self.values.append(np.random.beta(self.alpha, self.beta))


    def draw_first_offer(self):# Draw the first offer from a Binomial distribution    
        
        n = int(np.random.uniform(1000, 1000000))
        successes = sum(np.random.binomial(1, self.values[0], n))
        p_1 = successes/n
        self.succeses_first_ad = successes
        self.offers.append(p_1)
        sigma = mth.sqrt((p_1*(1-p_1)/n))
        self.sigmas.append(sigma)
        self.impressions_first = n
    
    
    def draw_other_offers(self):

        p_k = []
        # we skip the first value
        for i in range(1, len(self.values)) :
            n = int(self.N/(self.k-1))
            p_k = sum(np.random.binomial(1, self.values[i], n))/ n 
            self.offers.append(p_k)
            sigma = mth.sqrt((p_k*(1-p_k)/n))
            self.sigmas.append(sigma)
    
    def ShrinkageSelection(self, c):

        p_hat =[]
        for i in range(len(self.offers)):
            shrinked_offer = self.offers[i]-c*self.sigmas[i]
            p_hat.append(shrinked_offer)
        winner = p_hat.index(max(p_hat))
        return winner
    
    def BayesianBandits(self):

        alpha = []
        beta =[]
        ad_t = []
        y_t = []
        p_hat = []
        for i in range(self.k):
            alpha.append(self.alpha)
            beta.append(self.beta)
        thetas_hat = np.zeros(self.k)
        for j in range(self.N):
            for k in range(self.k):
                thetas_hat[k] = np.random.beta(alpha[k], beta[k])
            prob_selected = np.amax(thetas_hat)
            self.BayesianProbabilites.append(prob_selected)
            ad_selected = np.argmax(thetas_hat)
            self.bayes_arm_selected1.append(ad_selected)
            ad_t.append(ad_selected)
            y_t.append(np.random.binomial(1, self.values[ad_selected], 1))
            alpha[ad_selected] = alpha[ad_selected] + y_t[j]
            self.alphas.append(alpha[ad_selected])
            beta[ad_selected] = beta[ad_selected] + 1 -y_t[j]
            self.betas.append([beta[ad_selected]])
        for ad in range(self.k):
            probability = alpha[ad]/(alpha[ad] + beta[ad])
            p_hat.append(probability)
        winner = p_hat.index(max(p_hat))
        return winner
    '''
    function that update the parameters of the prior for the first ad 
    according to the observed sequence of succeses and failures during the
    first ad
    '''
    def UpdateFirstPrior(self, alphas, betas):
        alphas[0] = self.alpha + self.succeses_first_ad
        betas[0] = self.beta + self.impressions_first - self.succeses_first_ad
        return alphas, betas
    

    # our modified version of the bandits
    def BayesianBandits2(self):

        alpha_k = np.zeros(self.k)
        beta_k = np.zeros(self.k)
        ad_t = []
        y_t = []
        p_hat = []
        alpha_k, beta_k = self.UpdateFirstPrior(alpha_k, beta_k)
        for i in range(1, self.k):
            alpha_k[i] = self.alpha
            beta_k[i] = self.beta
        thetas_hat = np.zeros(self.k)
        for j in range(self.N):
            for k in range(self.k):
                thetas_hat[k] = np.random.beta(alpha_k[k], beta_k[k])
            self.BayesianProbabilites.append(thetas_hat)
            ad_selected = np.argmax(thetas_hat)
            self.bayes_arm_selected2.append(ad_selected)
            ad_t.append(ad_selected)
            y_t.append(np.random.binomial(n=1, p = self.values[ad_selected], size=1))
            alpha_k[ad_selected] = alpha_k[ad_selected] + y_t[j]
            #self.alphas.append(alpha_k[ad_selected])
            beta_k[ad_selected] = beta_k[ad_selected] + 1 -y_t[j]
            #self.betas.append([beta_k[ad_selected]])
        for ad in range(self.k):
            probability = alpha_k[ad]/(alpha_k[ad] + beta_k[ad])
            p_hat.append(probability)
        winner = p_hat.index(max(p_hat))
        self.alphas.append(alpha_k[winner])
        self.betas.append(beta_k[winner])
        return winner

    def RecordResults(self, winner):
        offer_values = self.values
        true_winner = offer_values.index(max(offer_values))
        if winner == true_winner:
            result = 1
        else:
            result = 0

        return result 
    
    def CalculateRegretShrink(self):#Calculate the total regret of each auction

        regret_t = []
        j = 1
        for n in range(1, self.k):
            l_t = (self.k-1)*(max(self.values)-self.values[j])
            regret_t.append(l_t)
            j += 1
        return sum(regret_t)
    
    def CalculateRegretBayes(self):
        regret_t = []
        for n in range(self.N):
            l_t = max(self.values)-self.values[self.bayes_arm_selected1[n]]
            regret_t.append(l_t)
        return sum(regret_t)
    
    def CalculateRegretBayes2(self):
        regret_t = []
        for n in range(self.N):
            l_t = max(self.values)-self.values[self.bayes_arm_selected2[n]]
            regret_t.append(l_t)
        return sum(regret_t)

 

# unit test
# intitialize the auction
test = Auctioneer(3, 100, 0.9809628968249227, 5.318600742609512)
test.draw_values()
test.draw_first_offer()
test.draw_other_offers()
# check values and offers
test.values
test.offers
# check selection methods
test.offers.index(max(test.offers))
test.ShrinkageSelection(0.5)
test.BayesianBandits()
test.BayesianBandits2()
# check parameters auction
test.succeses_first_ad
test.alpha
test.impressions_first
test.succeses_first_ad/test.impressions_first
# check regrets
test.CalculateRegretShrink()
test.CalculateRegretBayes()
test.CalculateRegretBayes2()



# Function that returns a matrix with the results for each policy
def SimulationComparison(k, N, alpha, beta, draws):
    
    c_values = np.linspace(0,1.5,12)
    # result matrix
    result_matrix = np.zeros(shape=(len(c_values),5))
    regret_matrix = np.zeros(shape=(draws, 3))
    naive_result = []
    shrink_result = np.zeros(shape=(len(c_values), draws))
    bandit_result = []
    bandit_result2 = []
    # we perform 10000 simulated auctions
    for i in range(draws):
        print(i)
        auction = Auctioneer(k, N, alpha, beta)
        auction.draw_values()
        auction.draw_first_offer()
        auction.draw_other_offers()
        naive_winner = auction.offers.index(max(auction.offers))
        # record result for the naive policy
        naive_result.append(auction.RecordResults(naive_winner))
        # record the result for each value of c
        j = 0
        for c in c_values:
            shrink_winner = auction.ShrinkageSelection(c)
            shrink_result[j, i] = auction.RecordResults(shrink_winner)
            j += 1
        bandit_winner = auction.BayesianBandits()
        # record result for the bandits policy
        bandit_result.append(auction.RecordResults(bandit_winner))
        # modified bandit
        bandit_winner2 = auction.BayesianBandits2()
        # record result for the bandits policy
        bandit_result2.append(auction.RecordResults(bandit_winner2))
        # record regrets
        regret_matrix[i, 0] = auction.CalculateRegretShrink()
        regret_matrix[i, 1] = auction.CalculateRegretBayes()
        regret_matrix[i, 2] = auction.CalculateRegretBayes2()
    means_naive = sum(naive_result)/draws
    #print(shrink_result)
    means_shrink = shrink_result.mean(axis=1)
    means_bandit = sum(bandit_result)/draws
    means_bandit2 = sum(bandit_result2)/draws
    for row in range(len(c_values)):
        result_matrix[row, 0] = c_values[row]
    for row in range(len(c_values)):
        result_matrix[row, 1] = means_naive
    for row in range(len(c_values)):
        result_matrix[row, 2] = means_shrink[row]
    for row in range(len(c_values)):
        result_matrix[row, 3] = means_bandit
    for row in range(len(c_values)):
        result_matrix[row, 4] = means_bandit2

            
    return result_matrix, regret_matrix
        
# unit test for this function
result_test, regret_test = SimulationComparison(8, 100, 1, 1, 1000)
np.mean(regret_test[0])
np.mean(regret_test[1])
np.mean(regret_test[2])
def RunExperiments(alpha, beta, draws):

    # running auctions with different parameters
    # and saving them to different .csv files
    for N in (100,1000):
        for k in range(3,9):
            data1, data2 = SimulationComparison(k, N, alpha, beta, draws)
            np.savetxt('results_' + str(N) + '_' + str(k), data1, delimiter=',')
            np.savetxt('regrets_' + str(N) + '_' + str(k), data2, delimiter=',')



test = RunExperiments(1, 1, 10000)


