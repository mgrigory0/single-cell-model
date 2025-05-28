import numpy as np
from scipy.stats import multivariate_normal

def initialize_parameters(X, K):
    n_samples, n_features = X.shape
    mu = X[np.random.choice(n_samples, K, replace=False)]
    pi = np.random.dirichlet(alpha=np.ones(K), size=1).flatten()
    Sigma = np.array([np.eye(n_features) for _ in range(K)])
    return mu, Sigma, pi

def regularize_covariance(cov, epsilon=1e-4):
    return cov + np.eye(cov.shape[0]) * epsilon

def e_step(X, mu, Sigma, pi):
    n_samples, n_features = X.shape
    K = len(pi)
    log_alpha = np.zeros((n_samples, K))
    
    for k in range(K):
        cov_matrix = regularize_covariance(Sigma[k])
        rv = multivariate_normal(mean=mu[k], cov=cov_matrix)
        log_pdf_values = rv.logpdf(X)
        log_alpha[:, k] = np.log(pi[k]) + log_pdf_values

    log_alpha_max = np.max(log_alpha, axis=1, keepdims=True)
    log_alpha -= log_alpha_max
    alpha = np.exp(log_alpha)
    alpha_sum = alpha.sum(axis=1)[:, np.newaxis]
    alpha_sum = np.maximum(alpha_sum, 1e-10)
    alpha /= alpha_sum

    if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
        raise ValueError("Alpha contains invalid values.")
    
    return alpha

def m_step(X, alpha):
    n_samples, n_features = X.shape
    K = alpha.shape[1]
    Nk = alpha.sum(axis=0)

    pi = Nk / n_samples
    pi = np.maximum(pi, 1e-10)
    pi /= np.sum(pi)

    mu = np.dot(alpha.T, X) / Nk[:, np.newaxis]

    Sigma = np.zeros((K, n_features, n_features))
    for k in range(K):
        diff = X - mu[k]
        poids = alpha[:, k][:, np.newaxis] * diff
        poids = np.dot(poids.T, diff)
        diag_covariance = np.diag(poids) / Nk[k]
        Sigma[k] = np.diag(diag_covariance)

        if np.any(np.isnan(Sigma[k])) or np.any(np.isinf(Sigma[k])):
            print(f"Invalid values in covariance matrix for cluster {k}")
    
    return mu, Sigma, pi

def log_likelihood(X, mu, Sigma, pi):
    n_samples = X.shape[0]
    n_clusters = len(pi)
    densities = []

    for k in range(n_clusters):
        try:
            densities.append(multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k]))
        except np.linalg.LinAlgError:
            Sigma[k] += np.eye(len(Sigma[k])) * 1e-6
            densities.append(multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k]))

    densities = np.array(densities).T
    weighted_densities = densities * pi
    ll = np.sum(np.log(np.sum(weighted_densities, axis=1)))
    return ll

def log_likelihood_complete(X, mu, Sigma, pi, alpha):
    log_densities = np.array([
        multivariate_normal.logpdf(X, mean=mu[k], cov=Sigma[k])
        for k in range(len(pi))
    ]).T
    weighted_log_densities = alpha * (np.log(pi) + log_densities)
    ll_comp = np.sum(weighted_log_densities)
    return ll_comp

def em_algorithm(X, K, max_iter=50, tol=1e-4):
    mu, Sigma, pi = initialize_parameters(X, K)
    log_likelihoods = []

    for iteration in range(max_iter):
        alpha = e_step(X, mu, Sigma, pi)
        mu, Sigma, pi = m_step(X, alpha)
        ll = log_likelihood_complete(X, mu, Sigma, pi, alpha)
        log_likelihoods.append(ll)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return mu, Sigma, pi, alpha

def find_best_initialization(X_ref, K, n_init=10, max_iter=100, tol=1e-6):
    best_ll_complete = -np.inf
    best_params = None
    for init in range(n_init):
        try:
            mu, Sigma, pi = initialize_parameters(X_ref, K)
            mu_est, Sigma_est, pi_est, alpha_est = em_algorithm(
                X_ref, K, max_iter=max_iter, tol=tol
            )
            ll_complete = log_likelihood_complete(X_ref, mu_est, Sigma_est, pi_est, alpha_est)
            if ll_complete > best_ll_complete:
                best_ll_complete = ll_complete
                best_params = {"mu": mu_est, "Sigma": Sigma_est, "pi": pi_est}
        except Exception as e:
            print(f"Initialization {init + 1}/{n_init} failed: {e}")
    if best_params is None:
        raise ValueError("No valid initialization was found.")
    return best_params

def em_algorithm_with_best_init(X, K, max_iter=100, tol=1e-6):
    try:
        best_params = find_best_initialization(X, K)
    except ValueError as e:
        print(e)
        return None, None, None, None
    
    mu, Sigma, pi = best_params["mu"], best_params["Sigma"], best_params["pi"]
    n_samples, n_features = X.shape
    log_likelihoods = []
    for iteration in range(max_iter):
        alpha = e_step(X, mu, Sigma, pi)
        mu, Sigma, pi = m_step(X, alpha)
        ll = log_likelihood(X, mu, Sigma, pi)
        ll_comp = log_likelihood_complete(X, mu, Sigma, pi, alpha)
        log_likelihoods.append(ll)
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    return mu, Sigma, pi, alpha





def compute_bic(X, num_param, log_likelihood):
    n_samples = X.shape[0]
    return (-2) * log_likelihood + np.log(n_samples) * num_param

def compute_aic(log_likelihood, num_param):
    return -2 * log_likelihood + 2 * num_param



def compute_icl(X, num_param, log_likelihood_comp):
    n_samples = X.shape[0]
    return (-2) * log_likelihood_comp + np.log(n_samples) * num_param



def em_algorithm_criterion(X, K, n_well, max_iter=100, tol=1e-6):
    try:
        best_params = find_best_initialization(X, K)
    except ValueError as e:
        print(e)
        return None, None, None, None
    
    mu, Sigma, pi = best_params["mu"], best_params["Sigma"], best_params["pi"]
    n_samples, n_features = X.shape
    log_likelihoods = []
    for iteration in range(max_iter):
        alpha = e_step(X, mu, Sigma, pi)
        mu, Sigma, pi = m_step(X, alpha)
        ll = log_likelihood(X, mu, Sigma, pi)
        ll_comp = log_likelihood_complete(X, mu, Sigma, pi, alpha)
        log_likelihoods.append(ll)
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    

    num_param = 2*K * n_features + (K-1)*n_well
    bic = compute_bic(X, num_param, ll)
    aic = compute_aic(ll, num_param)
    icl = compute_icl(X, num_param, ll_comp)

    print(f"K={K}, Log-likelihood={ll:.2f}, Log-likelihood Complete={ll_comp:.2f}, Num Params={num_param}, BIC={bic:.2f}, AIC={aic:.2f}, ICL={icl:.2f}")

    return mu, Sigma, pi, alpha, bic, aic, icl
