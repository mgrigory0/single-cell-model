import numpy as np
from scipy.stats import multivariate_normal

class EM_model:
    def __init__(self, X, n_well):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.n_well = n_well

    def initialize_parameters(self, K):
        np.random.seed(42)
        mu = self.X[np.random.choice(self.n_samples, K, replace=False)]
        pi = np.ones(K) / K
        Sigma = np.array([np.eye(self.n_features) for _ in range(K)])
        return mu, Sigma, pi

    def regularize_covariance(self, cov, epsilon=1e-4):
        return cov + np.eye(cov.shape[0]) * epsilon

    def e_step(self, mu, Sigma, pi):
        K = len(pi)
        log_alpha = np.zeros((self.n_samples, K))
        for k in range(K):
            cov_matrix = self.regularize_covariance(Sigma[k])
            rv = multivariate_normal(mean=mu[k], cov=cov_matrix)
            log_alpha[:, k] = np.log(pi[k]) + rv.logpdf(self.X)

        log_alpha -= np.max(log_alpha, axis=1, keepdims=True)
        alpha = np.exp(log_alpha)
        alpha /= np.maximum(alpha.sum(axis=1, keepdims=True), 1e-10)

        if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
            raise ValueError("Alpha contains invalid values.")

        return alpha

    def m_step(self, alpha):
        K = alpha.shape[1]
        Nk = alpha.sum(axis=0)

        pi = np.maximum(Nk / self.n_samples, 1e-10)
        pi /= pi.sum()

        mu = np.dot(alpha.T, self.X) / Nk[:, np.newaxis]

        Sigma = np.zeros((K, self.n_features, self.n_features))
        for k in range(K):
            diff = self.X - mu[k]
            poids = alpha[:, k][:, np.newaxis] * diff
            poids = np.dot(poids.T, diff)
            diag_cov = np.diag(poids) / Nk[k]
            Sigma[k] = np.diag(diag_cov)

        return mu, Sigma, pi

    def log_likelihood_complete(self, mu, Sigma, pi, alpha):
        log_densities = np.zeros((self.n_samples, len(pi)))
        for k in range(len(pi)):
            try:
                cov_matrix = self.regularize_covariance(Sigma[k])
                log_densities[:, k] = multivariate_normal.logpdf(self.X, mean=mu[k], cov=cov_matrix)
            except Exception:
                return -np.inf  # En cas de bug, on élimine cette init
        return np.sum(alpha * (np.log(pi) + log_densities))


    def log_likelihood(self, mu, Sigma, pi):
        densities = np.zeros((self.n_samples, len(pi)))
        for k in range(len(pi)):
            try:
                densities[:, k] = multivariate_normal.pdf(self.X, mean=mu[k], cov=Sigma[k])
            except np.linalg.LinAlgError:
                Sigma[k] += np.eye(self.n_features) * 1e-6
                densities[:, k] = multivariate_normal.pdf(self.X, mean=mu[k], cov=Sigma[k])

        return np.sum(np.log(np.sum(densities * pi, axis=1) + 1e-10))

    def compute_icl(self, num_param, log_likelihood_comp):
        return -2 * log_likelihood_comp + np.log(self.n_samples) * num_param

    def em_algorithm(self, K, max_iter=100, tol=1e-4):
        mu, Sigma, pi = self.initialize_parameters(K)
        log_likelihoods = []

        for _ in range(max_iter):
            alpha = self.e_step(mu, Sigma, pi)
            mu, Sigma, pi = self.m_step(alpha)
            ll = self.log_likelihood_complete(mu, Sigma, pi, alpha)
            log_likelihoods.append(ll)
            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break

        return mu, Sigma, pi, alpha, ll

    def em_algorithm_crit_with_best_init(self, K, n_init=10, max_iter=100, tol=1e-6):
        best_ll_complete = -np.inf
        best_params = None

        for _ in range(n_init):
            try:
                mu, Sigma, pi = self.initialize_parameters(K)
                for _ in range(max_iter):
                    alpha = self.e_step(mu, Sigma, pi)
                    mu, Sigma, pi = self.m_step(alpha)
                    ll_comp = self.log_likelihood_complete(mu, Sigma, pi, alpha)
                    if best_ll_complete < ll_comp:
                        best_ll_complete = ll_comp
                        best_params = (mu, Sigma, pi, alpha, ll_comp)
            except Exception:
                continue

        if best_params is None:
            raise ValueError("No valid initialization found.")

        mu, Sigma, pi, alpha, ll_comp = best_params
        ll = self.log_likelihood(mu, Sigma, pi)
        num_param = 2 * K * self.n_features + (K - 1) * self.n_well
        icl = self.compute_icl(num_param, ll_comp)

        print(f"K={K}, Log-likelihood={ll:.2f}, Log-likelihood Complete={ll_comp:.2f}, Num Params={num_param}, ICL={icl:.2f}")

        return mu, Sigma, pi, alpha, icl

    def find_optimal_K_by_icl(self, K_values, n_init=10, max_iter=100, tol=1e-6):
        icl_scores = []
        for K in K_values:
            print(f"Testing K={K}")
            _, _, _, _, icl = self.em_algorithm_crit_with_best_init(K, n_init=n_init, max_iter=max_iter, tol=tol)
            icl_scores.append(icl)
        optimal_K_icl = K_values[np.argmin(icl_scores)]
        print(f"\nOptimal K by ICL: {optimal_K_icl}")
        return icl_scores, optimal_K_icl
