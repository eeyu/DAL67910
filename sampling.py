import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def sample_from_pdf(target_density, dimension, size=500000):
    ## from https://github.com/abdulfatir/sampling-methods-numpy/
    burnin_size = int(size / 10.0)+100
    size += burnin_size
    x0 = np.zeros((1, dimension))
    xt = x0
    samples = []
    for i in tqdm(range(size)):
        xt_candidate = np.array([np.random.multivariate_normal(xt[0], np.eye(dimension))])
        accept_prob = (target_density(xt_candidate))/(target_density(xt))
        if (np.random.uniform(0, 1) < accept_prob).any():
            xt = xt_candidate
        samples.append(xt)
    samples = np.array(samples[burnin_size:])
    samples = np.reshape(samples, [samples.shape[0], dimension])
    return samples

class Distribution(ABC):
    @abstractmethod
    def sample(self, n) -> np.array:
        # returns n of x
        pass

    @abstractmethod
    def get_probability(self, x: np.array):
        pass


class LabelDistribution(ABC):
    @abstractmethod
    def _conditional_probability_x(self, x):
        pass

    def sample_labels(self, x) -> np.array:
        ## samples labels_y for values x
        probabilities = self._conditional_probability_x(x)

        size = len(x)
        noise = np.random.rand(size)
        labels = np.empty(size)
        labels[noise > probabilities] = 1
        labels[noise < probabilities] = 0
        return labels


class FastUniformOnRangeDistribution(Distribution):
    def __init__(self, range_min, range_max):
        ## region: the full space
        ## range: the desired distribution
        self.dimensions = len(range_min)
        self.range_min = range_min
        self.range_max = range_max

    def sample(self, n):
        # returns n of x
        return np.random.uniform(low=self.range_min, high=self.range_max, size=(n, self.dimensions))

    def get_probability(self, x: np.array):
        return range_pdf(x, self.range_min, self.range_max)

class FastGaussianDistribution(Distribution):
    def __init__(self, covar_matrix, mean):
        ## region: the full space
        ## range: the desired distribution
        self.dimensions = len(mean)
        self.mean = mean
        self.covar_matrix = covar_matrix

    def sample(self, n):
        # returns n of x
        return np.random.multivariate_normal(mean=self.mean, cov=self.covar_matrix, size=n)

    def get_probability(self, x: np.array):
        return gaussian_pdf(x, mean=self.mean, covar=self.covar_matrix)



class PDFDistribution(Distribution):
    def __init__(self, pdf, dimensions):
        self.pdf = pdf
        self.dimensions = dimensions

    def sample(self, n):
        return sample_from_pdf(target_density=self.pdf, dimension=self.dimensions, size=n)

    def get_probability(self, x: np.array):
        return self.pdf(x)


class UniformOnRangeDistribution(PDFDistribution):
    def __init__(self, range_min, range_max):
        ## region: the full space
        ## range: the desired distribution
        dimensions = len(range_min)
        pdf = lambda x: range_pdf(x, range_min=range_min, range_max=range_max)
        super(UniformOnRangeDistribution, self).__init__(pdf=pdf, dimensions=dimensions)

class GaussianDistribution(PDFDistribution):
    def __init__(self, covar, mean):
        ## region: the full space
        ## range: the desired distribution
        dimensions = len(mean)
        pdf = lambda x: gaussian_pdf(x, covar=covar, mean=mean)
        super(GaussianDistribution, self).__init__(pdf=pdf, dimensions=dimensions)

def range_pdf(x, range_min, range_max):
    # x: nxd
    # if (x-range_min < 0).any() or (x-range_max > 0).any():
    #     return 1e-9
    num_calc = len(x)
    range_mag = range_max - range_min
    pdf = 1.0 / np.prod(range_mag) * np.ones(num_calc)
    out_of_range = np.logical_or((x-range_min < 0), (x-range_max > 0))
    out_of_range = np.any(out_of_range, axis=1)
    pdf[out_of_range] = 1e-9
    return pdf


def gaussian_pdf(x, covar, mean):
    ## x is n x d for n samples, d dimensions
    dimensions = x.shape[1]
    # return scipy.stats.multivariate_normal(mean=mean, cov=covar)
    cov_inv = np.linalg.inv(covar)
    cov_det = np.linalg.det(covar)
    return 1.0 / (np.sqrt(cov_det * np.power(2*np.pi, dimensions))) * np.exp(-0.5 * (x - mean) @ cov_inv @ (x-mean).T)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RandomLabelDistribution(LabelDistribution):
    def _conditional_probability_x(self, x):
        size = len(x)
        return np.ones(size) * 0.5

class LinearLabelDistribution(LabelDistribution):
    def __init__(self, weights):
        ## weights need to be same dimensions as x
        self.weights = weights

    def _conditional_probability_x(self, x):
        probabilities = np.dot(x, self.weights)
        return sigmoid(probabilities)


if __name__=="__main__":
    def test_1d_gaussian_metro():
        input_range = 1.0
        num_samples = 10000
        dimension = 1

        distribution1 = lambda x: gaussian_pdf(x, np.array([[0.01]]), np.array([0.0]))
        samples = sample_from_pdf(distribution1, dimension=dimension, size=num_samples)

        plt.hist(samples, bins=10)
        plt.show()


    def test_2d_gaussian_metro():
        input_range = 1.0
        num_samples = 100000
        dimension = 2

        distribution1 = lambda x: gaussian_pdf(x, np.array([[0.01, 0], [0, 0.02]]), np.array([1.0, 0.0]))
        samples = sample_from_pdf(distribution1, dimension=dimension, size=num_samples)

        plt.hist2d(samples[:, 0], samples[:, 1], bins=20)
        plt.show()


    def test_2d_range_metro():
        input_range = 1.0
        num_samples = 100000
        dimension = 2

        mins = np.array([0.1, -0.7])
        maxs = np.array([0.9, 0.8])
        range_mins = np.ones((dimension)) * -input_range
        range_maxs = np.ones((dimension)) * input_range

        # cdf1 = lambda x: range_cdf(x, range_mins=range_mins, range_maxs=range_maxs, mins=mins, maxs=maxs)
        # samples = sample_from_cdf(cdf=cdf1, dimension=dimension, size=num_samples)

        distribution = UniformOnRangeDistribution(range_min=mins, range_max=maxs)
        samples = distribution.sample(num_samples)
        print(samples)

        plt.hist2d(samples[:, 0], samples[:, 1], bins=20,
                   range=[[-input_range, input_range], [-input_range, input_range]])
        plt.show()

    def test_2d_range_fast():
        input_range = 1.0
        num_samples = 100000
        dimension = 2

        mins = np.array([0.1, -0.7])
        maxs = np.array([0.9, 0.8])
        range_mins = np.ones((dimension)) * -input_range
        range_maxs = np.ones((dimension)) * input_range

        # cdf1 = lambda x: range_cdf(x, range_mins=range_mins, range_maxs=range_maxs, mins=mins, maxs=maxs)
        # samples = sample_from_cdf(cdf=cdf1, dimension=dimension, size=num_samples)

        distribution = FastUniformOnRangeDistribution(range_min=mins, range_max=maxs)
        samples = distribution.sample(num_samples)
        print(samples)

        plt.hist2d(samples[:, 0], samples[:, 1], bins=20,
                   range=[[-input_range, input_range], [-input_range, input_range]])
        plt.show()

    def test_label_distribution_random():
        input_range = 1.0
        num_samples = 100000
        dimension = 30

        mins = np.ones(dimension) * -input_range
        maxs = np.ones(dimension) * input_range

        distribution = FastUniformOnRangeDistribution(range_min=mins, range_max=maxs)
        x = distribution.sample(num_samples)

        label_distribution = RandomLabelDistribution()
        y = label_distribution.sample_labels(x)

        print(np.sum(y)/num_samples)
        print("expected", "0")

    def test_label_distribution_biased():
        input_range = 1.0
        num_samples = 1000
        dimension = 2

        mins = np.ones(dimension) * -input_range
        maxs = np.ones(dimension) * input_range

        # distribution = FastUniformOnRangeDistribution(range_min=mins, range_max=maxs)
        distribution = FastGaussianDistribution(mean=np.array([-0.5, 0]), covar_matrix=np.array([[0.1,0],[0,0.1]]))
        x = distribution.sample(num_samples)

        # label_distribution = RandomLabelDistribution()
        label_distribution = LinearLabelDistribution(weights=np.array([10, 0]))

        # y = random_label_distribution(samples)
        y = label_distribution.sample_labels(x)
        positive = x[y>0]
        negative = x[y<0]

        plt.scatter(positive[:,0], positive[:,1], c='r', alpha=0.5)
        plt.scatter(negative[:,0], negative[:,1], c='b', alpha=0.5)
        plt.show()

    test_label_distribution_biased()



# def sample_from_cdf(cdf, dimension, size=100):
#     ## TODO need to invert the cdf
#     unif_samples = np.random.rand(size, dimension)
#     transformed_samples = cdf(unif_samples)
#     return transformed_samples

# def pdf_from_cdf(cdf, x, domain_min, domain_max):
#     # pdf is the slope of cdf
#     domain_mag = domain_max - domain_min
#
#     dx = domain_mag / 100.0
#     upper = cdf(x + dx)
#     lower = cdf(x)
#     slope = (upper - lower) / dx
#     return slope


# class InvCDFDistribution(Distribution):
#     def __init__(self, inv_cdf, domain_min, domain_max):
#         self.inv_cdf = inv_cdf ## TODO make sure to pass cdf or inverse cdf appropriately
#         self.dimensions = len(domain_min)
#         self.pdf = lambda x: pdf_from_cdf(cdf=self.inv_cdf, x=x, domain_min=domain_min, domain_max=domain_max) #TODO this is wrong
#
#     def sample(self, n):
#         return sample_from_cdf(cdf=self.inv_cdf, dimension=self.dimensions, size=n)
#
#     def get_probability(self, x: np.array):
#         return self.pdf(x)

# def range_inv_cdf(p, range_min, range_max):
#     ## x from 0 to 1
#     ## range: the full distribution
#     # mins/maxs: the desired region
#     return range_min + (range_max - range_min) * (p)
