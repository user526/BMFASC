#coding=utf-8
import numpy as np
import os
from collections import Counter
from sklearn import metrics
import time
import sys, traceback
from sklearn.semi_supervised import label_propagation
from scipy.linalg import solve_triangular
import global_variables
from scipy.sparse import csr_matrix, vstack

def high_fidelity_experimental(q, c):
    return global_variables.all_hfs[q][c]


def low_fidelity_experimental(q, c, seed_set_high):
    return global_variables.all_lfs[q][c]


def retrieve_label_propagation(Lambda_res, lam_h, seed_set_high, n_neighbours, restrict_positive_hfs=False, q=None):
    if Lambda_res < lam_h:
        return None
      
    labels = np.zeros(len(global_variables.embedding)) - 1
    remaining_elements = set(range(len(global_variables.embedding))) - set([v for v, s in seed_set_high])
    if len(remaining_elements) == 0:
        return None # assuming that undiscovered low fidelities are depleted before the high ones!
    remaining_elements = np.array(list(remaining_elements))
    if restrict_positive_hfs:
        remaining_elements = np.setdiff1d(remaining_elements, np.where(global_variables.all_hfs[q] > 0)[0])
    
    threshold = np.percentile([s for _, s in seed_set_high], 90) - 0.05

    labels = np.zeros(len(global_variables.embedding)) - 1
    for i, s in seed_set_high:
        if s > threshold:
            labels[i] = 1
        else:
            labels[i] = 0
    lp_model = label_propagation.LabelSpreading(kernel = 'knn', n_neighbors = n_neighbours, max_iter=10)
    lp_model.fit(global_variables.embedding, labels)
    class_1 = np.where(lp_model.classes_ == 1)[0]
    class_1_ind = class_1[0]
    v = remaining_elements[np.argmax(lp_model.label_distributions_[remaining_elements, class_1_ind])]
    
    return v

def gaussian_posterior(mu_prior, cov_prior, X, y, sigma_n, return_covariance_matrix = False):
    cov_y = (X.dot((X.dot(cov_prior)).T)).T + sigma_n**2 * np.eye(len(y)) # note (X.dot((X.dot(cov_prior)).T)).T == X.dot(cov_prior).dot(X.T) in dense format
    L_y = np.linalg.cholesky(cov_y)

    minus_H = solve_triangular(L_y, y - X.dot(mu_prior), lower = True)

    minus_R = solve_triangular(L_y, X.dot(cov_prior), lower = True)
    v_posterior = np.diag(cov_prior) - np.einsum('ik,ki->i', minus_R.T, minus_R)
    mu_posterior = mu_prior + (minus_R.T).dot(minus_H)
    if return_covariance_matrix:
        return mu_posterior, cov_prior - minus_R.T.dot(minus_R)
    else:
        return mu_posterior, v_posterior # variance - diag of cov


import matplotlib.pyplot as plt
def mf_gaussian_posterior_diff_high(mu_prior_low, cov_prior_low,
                          mu_prior_diff, cov_prior_diff, 
                          X_low, y_low, X_high, y_high, 
                          sigma_n_low, sigma_n_high, 
                          rho_prior):
    mu_posterior_low, _ = gaussian_posterior(mu_prior_low, cov_prior_low, X_low, y_low, sigma_n_low)
    sigma_n_diff = np.sqrt(np.max([sigma_n_high**2 - rho_prior**2 * sigma_n_low**2, 1e-4]))#to ensure positive definiteness
    C_d_h = (X_high.dot((X_high.dot(cov_prior_diff)).T)).T + sigma_n_diff**2 * np.eye(len(y_high)) #csr_sparse
    L_y = np.linalg.cholesky(C_d_h)
    
    Y_l_h = X_high.dot(mu_posterior_low)
    minus_L_y_Y_l_h = solve_triangular(L_y, Y_l_h, lower=True)
    minus_C_d_hh_Y_l_h = solve_triangular(L_y.T, minus_L_y_Y_l_h, lower=False)
    rho_posterior = y_high.dot(minus_C_d_hh_Y_l_h) / Y_l_h.dot(minus_C_d_hh_Y_l_h)

    C_d_hh = (X_high.dot((X_high.dot(cov_prior_diff)).T)).T + sigma_n_diff**2 * np.eye(len(y_high)) #csr_sparse
    C_l_hh = (X_high.dot((X_high.dot(cov_prior_low)).T)).T + sigma_n_low**2 * np.eye(len(y_high)) #csr_sparse
    C_l_ll = (X_low.dot((X_low.dot(cov_prior_low)).T)).T + sigma_n_low**2 * np.eye(len(y_low)) #csr_sparse
    C_l_lh = (X_high.dot((X_low.dot(cov_prior_low)).T)).T + sigma_n_low**2 * X_low.dot(X_high.T) #csr_sparse
    
    C = np.vstack((np.hstack((C_l_ll,rho_posterior * C_l_lh)),
                  np.hstack((rho_posterior * C_l_lh.T,rho_posterior**2 * C_l_hh + C_d_hh))))
    c = np.vstack((rho_posterior * X_low.dot(cov_prior_low),
                   rho_posterior**2 * X_high.dot(cov_prior_low) + X_high.dot(cov_prior_diff)))
    
    L_y = np.linalg.cholesky(C)
    
    
    mu_prior_high = rho_posterior**2 * mu_prior_low + mu_prior_diff
    v_prior_high = rho_posterior**2 * np.diag(cov_prior_low) + np.diag(cov_prior_diff)
    y = np.hstack((y_low, y_high))
    minus_H = solve_triangular(L_y, y - vstack((X_low, X_high)).dot(mu_prior_high), lower = True)
    minus_R = solve_triangular(L_y, c, lower = True)
    v_posterior_high = v_prior_high - np.einsum('ik,ki->i', minus_R.T, minus_R)
    mu_posterior_high = mu_prior_high + (minus_R.T).dot(minus_H)
    return rho_posterior, mu_posterior_high, v_posterior_high   
    
def run_test_labelprop(Q, q, Lambda, lambda_high, seed_set_high, n_neighbours, restrict_positive_hfs=False):
    logger = {'iter_data':[], 'params':{}}
    logger['params'] = {'lambda_high':lambda_high,
                        'Lambda':Lambda,
                        '_q':q,
                        'Q':Q[q]}
    total_cost = 0
    iteration = 0
    start_time = time.time()
    while total_cost < Lambda:
        print('.', end = '')
        
        logger['iter_data'].append({
            'iteration':iteration, 
                       'seed_set_high':list(seed_set_high),
                       'time':time.time() - start_time,
                       'total_cost':total_cost
                       })

        iteration += 1    
        c = retrieve_label_propagation(Lambda - total_cost, lambda_high, seed_set_high, n_neighbours, restrict_positive_hfs, q)
        if c is None:
            break
        score = high_fidelity_experimental(q, c)
        seed_set_high.append((c, score))
        total_cost += lambda_high
    return logger


def run_test_gpsopttt(Q, q, Lambda, lambda_high, seed_set_high, k, alpha, obs_sigma = 0.01):
    logger = {'iter_data':[], 'params':{}}
    logger['params'] = {
                        'lambda_high':lambda_high,
                        'Lambda':Lambda,
                        '_q':q,
                        'Q':Q[q]}
    
    n = len(global_variables.V_sims)

    omega_0 = 0.01
    assert( np.all(np.abs(global_variables.V_sims - global_variables.V_sims.T) < 1e-6) ) # check that adjacency matrix is symmetric
    L = np.diag(np.sum(global_variables.V_sims, axis=1) + omega_0) - global_variables.V_sims
    C_prior = np.linalg.inv(L)
    
    
    mu_prior = 0.05 # prior positive rate

    T = Lambda + len(seed_set_high)
    selected = np.zeros(T, dtype = np.int) + np.nan
    
    y = np.array(global_variables.all_hfs[q])
    start_points = [c for c, s in seed_set_high]
    t0 = len(start_points)
    last_ind = np.array(start_points)
    selected[:t0] = last_ind
    mu_t = mu_prior * np.ones(n)
    C_t = C_prior
    
    total_cost = 0
    iteration = 0
    start_time = time.time()
    for t in range(t0, T):
        
        logger['iter_data'].append({'iteration':iteration, 
                       'seed_set_high':list(seed_set_high),
                       'time':time.time() - start_time,
                       'total_cost':total_cost})
        iteration += 1    

        if len(last_ind) > 0:
            m = len(last_ind)
            X = csr_matrix((np.ones(m), (np.arange(m), last_ind)), shape = (m, n)) # changed to csr_matrix
            mu_t, C_posterior = gaussian_posterior(mu_prior * np.ones(n), C_prior, X, y[last_ind], obs_sigma, return_covariance_matrix = True)

            
        sopt_noiseless = C_posterior.sum(axis = 1) / np.sqrt(np.diag(C_posterior));
        tt_clip = k*np.sqrt(np.diag(C_posterior));

        score_exploration = np.minimum(tt_clip, sopt_noiseless);

        score_exploitation = mu_t
            
        score = score_exploitation + alpha * score_exploration
        score[selected[:t].astype(np.int)] = -np.inf # to avoid duplicates

        cur_last_ind = np.argmax(score)
        selected[t] = cur_last_ind
        
        seed_set_high.append((cur_last_ind, y[cur_last_ind]))
        last_ind = np.hstack((last_ind, np.array([cur_last_ind])))
        total_cost += lambda_high
    return logger


def run_test_gplapl(Q, q, Lambda, lambda_high, seed_set_high, alpha, obs_sigma = 0.01, restrict_positive_hfs=False):
    logger = {'iter_data':[], 'params':{}}
    logger['params'] = {
                        'lambda_high':lambda_high,
                        'Lambda':Lambda,
                        '_q':q,
                        'Q':Q[q]}
    
    n = len(global_variables.V_sims)

    omega_0 = 0.01
    assert( np.all(np.abs(global_variables.V_sims - global_variables.V_sims.T) < 1e-6) ) # check that adjacency matrix is symmetric
    L = np.diag(np.sum(global_variables.V_sims, axis=1) + omega_0) - global_variables.V_sims
    C_prior = np.linalg.inv(L)
    
    
    mu_prior = 0.05 # prior positive rate

    T = Lambda + len(seed_set_high)
    selected = np.zeros(T, dtype = np.int) + np.nan
    
    y = np.array(global_variables.all_hfs[q])
    start_points = [c for c, s in seed_set_high]
    t0 = len(start_points)
    last_ind = np.array(start_points)
    selected[:t0] = last_ind
    mu_t = mu_prior * np.ones(n)
    C_t = C_prior
    
    total_cost = 0
    iteration = 0
    start_time = time.time()
    for t in range(t0, T):
        #print '.',
        
        logger['iter_data'].append({'iteration':iteration, 
                       'seed_set_high':list(seed_set_high),
                       'time':time.time() - start_time,
                       'total_cost':total_cost})
        iteration += 1    
        # update posterior
        if len(last_ind) > 0:
            m = len(last_ind)
            X = csr_matrix((np.ones(m), (np.arange(m), last_ind)), shape = (m, n)) # changed to csr_matrix
            mu_t, v_t = gaussian_posterior(mu_prior * np.ones(n), C_prior, X, y[last_ind], obs_sigma)

        score_exploration = np.sqrt(v_t)
        score_exploitation = mu_t
            
        score = score_exploitation + alpha * score_exploration
        score[selected[:t].astype(np.int)] = -np.inf # to avoid duplicates
        if restrict_positive_hfs:
            score[np.where(global_variables.all_hfs[q] == 0)[0]] = -np.inf
        # retrieve 
        cur_last_ind = np.argmax(score)
        selected[t] = cur_last_ind
        
        seed_set_high.append((cur_last_ind, y[cur_last_ind]))
        last_ind = np.hstack((last_ind, np.array([cur_last_ind])))
        total_cost += lambda_high
    return logger



def run_test_mfgplapl(Q, q, Lambda, lambda_high, lambda_low, seed_set_high, seed_set_low, alpha_high, alpha_low, low_per_high, obs_sigma = 0.01, restrict_positive_hfs=False):
    logger = {'iter_data':[], 'params':{}}
    logger['params'] = {'lambda_low':lambda_low,
                        'lambda_high':lambda_high,
                        'Lambda':Lambda,
                        '_q':q,
                        'Q':Q[q]}
    total_cost = 0
    iteration = 0
    start_time = time.time()
    
    
    
    n = len(global_variables.V_sims)

    T = Lambda + len(seed_set_high)
    selected = np.zeros(T, dtype = np.int) + np.nan
    selected_low = np.zeros(T*low_per_high, dtype = np.int) + np.nan
        
    omega_0 = 0.01
    assert( np.all(np.abs(global_variables.V_sims - global_variables.V_sims.T) < 1e-6) ) # check that adjacency matrix is symmetric
    L = np.diag(np.sum(global_variables.V_sims, axis=1) + omega_0) - global_variables.V_sims
    C_prior = np.linalg.inv(L)

    #obs_sigma = 0.01 # observation noise
    mu_prior = 0.05 # prior positive rate
    rho_prior = 0.5 # prior correlation between low and high fidelities
    start_points = [c for c, s in seed_set_high]
    
    t0 = len(start_points)

    y = np.array(global_variables.all_hfs[q])
    last_ind_low = np.array(start_points)
    last_ind_high = np.array(start_points)
    selected[:t0] = last_ind_high
    selected_low[:t0] = last_ind_low

    mu_t_high = mu_prior * np.ones(n)
    C_t_high = C_prior

    mu_t_i_low = mu_prior * np.ones(n)
    C_t_i_low = C_prior

    mu_t_diff = np.zeros(n)
    C_t_diff = C_prior

    cnt_selected_low = len(start_points)

    for t in range(t0, T):
        logger['iter_data'].append({'iteration':iteration, 'rho_prior':rho_prior,
                   'seed_set_high':list(seed_set_high),
                   'seed_set_low':list(seed_set_low),
                   'time':time.time() - start_time,
                   'total_cost':total_cost})
        iteration += 1    
        #print '.',
        # update low fidelity posteriors
        for i in range(low_per_high):
            last_ind_low = np.array([x[0] for x in seed_set_low])
            mu_t_i_low = mu_prior * np.ones(n)
            C_t_i_low = C_prior
            
             # update low fidelity posterior
            if len(last_ind_low) > 0:

                # update y_low[last_ind_low]
                y_low = np.array([low_fidelity_experimental(q, ind, seed_set_high) 
                                               for ind in last_ind_low])
                
                m = len(last_ind_low)
                X_low = csr_matrix((np.ones(m), (np.arange(m), last_ind_low)), shape = (m, n)) # changed to csr_matrix
                mu_t_i_low, v_t_i_low = gaussian_posterior(mu_t_i_low, C_t_i_low, X_low, y_low, obs_sigma)
            score_exploration = np.sqrt(v_t_i_low)
            score_exploitation = mu_t_i_low

            if rho_prior > 0:
                score = score_exploitation + alpha_low * score_exploration
                score[selected_low[cnt_selected_low - len(seed_set_low):cnt_selected_low].astype(np.int)] = -np.inf # to avoid duplicates
                last_ind_low = np.argmax(score)
            else:
                score = score_exploitation - alpha_low * score_exploration
                score[selected_low[cnt_selected_low - len(seed_set_low):cnt_selected_low].astype(np.int)] = +np.inf # to avoid duplicates
                last_ind_low = np.argmin(score)

            selected_low[cnt_selected_low] = last_ind_low
            seed_set_low.append((last_ind_low, low_fidelity_experimental(q, last_ind_low, seed_set_high) ))
            #if len(seed_set_low) > 3*len(seed_set_high):
            #    seed_set_low = seed_set_low[1:]
            cnt_selected_low += 1
            last_ind_low = np.array([last_ind_low])
            logger['iter_data'].append({'iteration':iteration, 'rho_prior':rho_prior,
                   'seed_set_high':list(seed_set_high),
                   'seed_set_low':list(seed_set_low),
                   'time':time.time() - start_time,
                   'total_cost':total_cost})
            iteration += 1    
        
        last_ind_high = np.array([x[0] for x in seed_set_high])
        
        
        mu_t_diff = np.zeros(n)
        C_t_diff = C_prior
        
        mu_t_i_low = mu_prior * np.ones(n)
        C_t_i_low = C_prior

            
        # update high_fidelity posterior
        y_high = y[last_ind_high]
        if len(last_ind_high) > 0:
            m = len(last_ind_high)
            X_high = csr_matrix((np.ones(m), (np.arange(m), last_ind_high)), shape = (m, n)) # changed to csr_matrix
            rho_prior, mu_t_high, v_t_high = mf_gaussian_posterior_diff_high(mu_t_i_low, C_t_i_low, 
                                                        mu_t_diff, C_t_diff,
                                                        X_low, y_low,
                                                        X_high, y_high, 
                                                        obs_sigma, obs_sigma,
                                                        rho_prior)

        
        score_exploration = np.sqrt(v_t_high)
        score_exploitation = mu_t_high

        score = score_exploitation + alpha_high * score_exploration
        score[selected[:t].astype(np.int)] = -np.inf # to avoid duplicates
        if restrict_positive_hfs:
            score[np.where(global_variables.all_hfs[q] == 0)[0]] = -np.inf
        last_ind_high = np.argmax(score)

        selected[t] = last_ind_high
        seed_set_high.append((last_ind_high, y[last_ind_high]))
        last_ind_high = np.array([last_ind_high])
        
        total_cost += lambda_high
    return logger

def run_test_mfgpucb(Q, q, Lambda, lambda_high, lambda_low, seed_set_high, seed_set_low, alpha0, d, obs_sigma = 0.01, restrict_positive_hfs=False):
    logger = {'iter_data':[], 'params':{}}
    logger['params'] = {'d':d,
                        'lambda_high':lambda_high,
                        'lambda_low':lambda_low,
                        'Lambda':Lambda,
                        '_q':q,
                        'Q':Q[q]}
    
    n = len(global_variables.V_sims)

    omega_0 = 0.01
    assert( np.all(np.abs(global_variables.V_sims - global_variables.V_sims.T) < 1e-6) ) # check that adjacency matrix is symmetric
    L = np.diag(np.sum(global_variables.V_sims, axis=1) + omega_0) - global_variables.V_sims
    C_prior = np.linalg.inv(L)
    
    
    mu_prior = 0.05 # prior positive rate
    #obs_sigma = 0.01 # observation noise

    selected = []
    
    y = np.array(global_variables.all_hfs[q])
    y_low = np.array(global_variables.all_lfs[q])
    start_points = [c for c, s in seed_set_high]
    t0 = len(start_points)
    last_ind = np.array(start_points)
    last_ind_low = np.array(start_points)
    
    selected = last_ind.tolist()
    selected_low = last_ind_low.tolist()
    mu_t = mu_prior * np.ones(n)
    C_t = C_prior

    mu_t_low = mu_prior * np.ones(n)
    C_t_low = C_prior
    
    total_cost = 0
    iteration = 0
    start_time = time.time()

    zeta = (np.max(y[selected]) - np.min(y[selected]))*0.01
    gamma = float(zeta)
    stuck_rate = lambda_high/lambda_low
    stuck_cnt = 0
    t = len(selected) + len(selected_low)

    while (total_cost + lambda_high <= Lambda):
        
        alpha_t = alpha0*np.sqrt(0.2*d*np.log(2*t))

        logger['iter_data'].append({'iteration':iteration, 
                       'seed_set_high':list(seed_set_high),
                       'seed_set_low':list(seed_set_low),
                       'time':time.time() - start_time,
                       'total_cost':total_cost,
                       'alpha_t':alpha_t})
        iteration += 1    
        # update posterior
        if len(last_ind) > 0:
            m = len(last_ind)
            X = csr_matrix((np.ones(m), (np.arange(m), last_ind)), shape = (m, n)) # changed to csr_matrix
            mu_t, v_t = gaussian_posterior(mu_prior * np.ones(n), C_prior, X, y[last_ind], obs_sigma)

        if len(last_ind_low) > 0:
            m = len(last_ind_low)
            X = csr_matrix((np.ones(m), (np.arange(m), last_ind_low)), shape = (m, n)) # changed to csr_matrix
            mu_t_low, v_t_low = gaussian_posterior(mu_prior * np.ones(n), C_prior, X, y_low[last_ind_low], obs_sigma)
    

        score_exploration = np.sqrt(v_t)
        score_exploration_low = np.sqrt(v_t_low)
        score_exploitation = mu_t
        score_exploitation_low = mu_t_low
            
        
        score = np.minimum(score_exploitation + alpha_t * score_exploration, 
                           score_exploitation_low + alpha_t * score_exploration_low + zeta)
        score[np.array(selected).astype(np.int)] = -np.inf # to avoid duplicates
        if restrict_positive_hfs:
            score[np.where(global_variables.all_hfs[q] == 0)[0]] = -np.inf
        # retrieve 
        for cur_last_ind in np.argsort(score)[::-1]:
            update_high = False
            if cur_last_ind not in selected_low and\
                alpha_t * score_exploration_low[cur_last_ind] >= gamma:
                if stuck_cnt > stuck_rate:
                    #while alpha_t * score_exploration_low[cur_last_ind] >= gamma:
                    #    gamma *= 2
                    gamma *= 2
                    update_high = True
                else:
                    selected_low.append(cur_last_ind)
                    seed_set_low.append((cur_last_ind, y_low[cur_last_ind]))
                    last_ind_low = np.hstack((last_ind_low, np.array([cur_last_ind])))
                    total_cost += lambda_low
                    stuck_cnt += 1
                    break
            else:
                update_high = True

            if update_high:
                if np.abs(y[cur_last_ind] - mu_t_low[cur_last_ind]) > zeta:
                    if cur_last_ind not in selected_low:
                        selected_low.append(cur_last_ind)
                        seed_set_low.append((cur_last_ind, y_low[cur_last_ind]))
                        last_ind_low = np.hstack((last_ind_low, np.array([cur_last_ind])))
                        total_cost += lambda_low
                    if np.abs(y[cur_last_ind] - y_low[cur_last_ind]) > zeta:
                        violation = np.abs(y[cur_last_ind] - y_low[cur_last_ind]) - zeta
                        zeta += 2*violation

                selected.append(cur_last_ind)
                seed_set_high.append((cur_last_ind, y[cur_last_ind]))
                last_ind = np.hstack((last_ind, np.array([cur_last_ind])))
                total_cost += lambda_high
                stuck_cnt = 0
                break

        t += 1
    return logger