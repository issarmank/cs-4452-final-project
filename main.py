from test_hpo import TestHpo

if __name__ == "__main__":
    tester = TestHpo()
    tester.test_grid_search_performance()
    tester.test_random_search_performance()
    tester.test_successive_halving_performance()
    tester.test_bayesian_optimization()