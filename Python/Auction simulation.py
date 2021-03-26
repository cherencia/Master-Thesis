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
        
    
    def draw_values(self): # Draw the theta values from a beta distribution
        for i in range(self.k):
            self.values.append(np.random.beta(self.alpha, self.beta))


    def draw_first_offer(self):# Draw the first offer from a Binomial distribution    
        
        n = int(np.random.uniform(1000, 1000000))
        p_1 = sum(np.random.binomial(1, self.values[0], n))/n
        self.offers.append(p_1)
        sigma = mth.sqrt((p_1*(1-p_1)/n))
        self.sigmas.append(sigma)
    
    
    def draw_other_offers(self):

        p_k = []
        # we skip the first value
        for i in range(1, len(self.values)) :
            n = int(np.random.uniform(1, self.N))
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
            ad_selected = np.argmax(thetas_hat)
            ad_t.append(ad_selected)
            y_t.append(np.random.binomial(1, self.values[ad_selected], 1))
            alpha[ad_selected] = alpha[ad_selected] + y_t[j]
            beta[ad_selected] = beta[ad_selected] + 1 -y_t[j]
        for ad in range(self.k):
            probability = alpha[ad]/(alpha[ad] + beta[ad])
            p_hat.append(probability)
        winner = p_hat.index(max(p_hat))
        return winner

    def RecordResults(self, winner):
        offer_values = self.values
        true_winner = offer_values.index(max(offer_values))
        if winner == true_winner:
            result = 1
        else:
            result = 0

        return result 


test = Auctioneer(3, 1000, 0.9809628968249227, 5.318600742609512)
test.draw_values()
test.draw_first_offer()
test.draw_other_offers()
shrink_winner = test.ShrinkageSelection(1)
bandit_winner = test.BayesianBandits()
record = test.RecordResults(bandit_winner)
record2 = test.RecordResults(shrink_winner)

# Function that returns a matrix with the results for each policy
def Simulation(k, N, alpha, beta, draws):
    
    c_values = np.linspace(0,1,10)
    # result matrix
    result_matrix = np.zeros(shape=(len(c_values),4))
    naive_result = []
    shrink_result = np.zeros(shape=(len(c_values), draws))
    bandit_result = []
    # we perform 10000 simulated auctions
    for i in range(draws):
        print(i)
        auction = Auctioneer(k, N, alpha, beta)
        auction.draw_values()
        #print(auction.values)
        auction.draw_first_offer()
        auction.draw_other_offers()
        #print(auction.offers)
        #print(auction.sigmas)
        naive_winner = auction.offers.index(max(auction.offers))
        # record result for the naive policy
        naive_result.append(auction.RecordResults(naive_winner))
        # record the result for each value of c
        j = 0
        for c in c_values:
            shrink_winner = auction.ShrinkageSelection(c)
            #print(shrink_winner)
            shrink_result[j, i] = auction.RecordResults(shrink_winner)
            j += 1
        bandit_winner = auction.BayesianBandits()
        # record result for the bandits policy
        bandit_result.append(auction.RecordResults(bandit_winner))
    means_naive = sum(naive_result)/draws
    #print(shrink_result)
    means_shrink = shrink_result.mean(axis=1)
    means_bandit = sum(bandit_result)/draws
    for row in range(len(c_values)):
        result_matrix[row, 0] = c_values[row]
    for row in range(len(c_values)):
        result_matrix[row, 1] = means_naive
    for row in range(len(c_values)):
        result_matrix[row, 2] = means_shrink[row]
    for row in range(len(c_values)):
        result_matrix[row, 3] = means_bandit

            
    return result_matrix
        
wasa = Simulation(10, 100, 0.9809628968249227, 5.318600742609512, 100)


