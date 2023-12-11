import numpy as np
import sampling


## https://stackoverflow.com/questions/14071704/integrating-a-multidimensional-integral-in-scipy

def mc_function_expectation(distribution: sampling.Distribution,
                            function,  # x: nxR^d -> nxR
                            num_samples: int
                            ):
    samples = distribution.sample(num_samples)
    print(samples)
    return np.sum(function(samples)) / num_samples

def get_inner_product_function(distribution: sampling.Distribution, f, g, num_samples=100000):
    multiple = lambda x: f(x)*g(x)
    return mc_function_expectation(distribution=distribution, function=multiple, num_samples=num_samples)

def get_norm_function(distribution: sampling.Distribution, f, num_samples=100000):
    return get_inner_product_function(distribution, f, f, num_samples)

if __name__=="__main__":
    def test_mc_integration1D():
        distribution = sampling.UniformOnRangeDistribution(range_min=np.array([-1]), range_max=np.array([1]))
        function = lambda x: x
        num_samples = 100000
        print(mc_function_expectation(distribution, function, num_samples))
        print("expected: ", "0")

        distribution = sampling.UniformOnRangeDistribution(range_min=np.array([0]), range_max=np.array([1]))
        function = lambda x: x
        num_samples = 10000
        print(mc_function_expectation(distribution, function, num_samples))
        print("expected: ", "0.5")

    def test_mc_integration2D():
        distribution = sampling.UniformOnRangeDistribution(range_min=np.array([-1, -1]), range_max=np.array([1, 1]))
        function = lambda x: x
        num_samples = 100000
        print(mc_function_expectation(distribution, function, num_samples))
        print("expected: ", "0")

        distribution = sampling.UniformOnRangeDistribution(range_min=np.array([0, 0]), range_max=np.array([1, 1]))
        function = lambda x: np.inner(np.array([0.5, 0.5]), x)
        num_samples = 10000
        print(mc_function_expectation(distribution, function, num_samples))
        print("expected: ", "0.5")

    test_mc_integration2D()


# def get_inner_product_function_over_distribution(distribution: sampling.Distribution):
#     return lambda f,g: integrate.quad(lambda t: f(t)*g(t), min, max)[0]
#
# class ComputeOrthogonalBasis1D:
#     def __init__(self, function):
#         self.function = function
#
#
# func_innprd=lambda f,g: integrate.quad(lambda t: f(t)*g(t),-np.pi,np.pi)[0]
# func_norm=lambda f: np.sqrt(func_innprd(f,f))
#
# polynomial_degree=13
# def orig_basis_vec(i): return lambda t: t**i
#
# orig_bis=[orig_basis_vec(i) for i in range(polynomial_degree+1)]
# orth_bis={}
# ips={}
#
# def compute_inner_prods(k=5,degree=5):
#     for i in range(k,degree+1):
#         xi=orig_bis[i]
#         ek=orth_bis[k]
#         ips[(i,k)]=func_innprd(xi,ek)
#     return
#
# def gram_schmidt(k=5,degree=5):
#     fk=lambda t: orig_bis[k](t)-np.sum([ips[(k,i)] * orth_bis[i](t) for i in range(k)])
#     nfk=func_norm(fk)
#     ek=lambda t: (1/nfk) * fk(t)
#     orth_bis[k]=ek
#     compute_inner_prods(k=k,degree=degree)
#     return ek
#
# for i in range(polynomial_degree+1): gram_schmidt(k=i,degree=polynomial_degree)
#
# def compute_PUv_coeffs(v,degree):
#     return [func_innprd(v,orth_bis[i]) for i in range(degree)]
#
# def PUv(t, PUv_coefficients):
#     return np.sum(PUv_coefficients[i]*orth_bis[i](t) for i in range(len(PUv_coefficients)))
#
# def graph(funct, x_range, cl='r--'):
#     y_range=[]
#     for x in x_range:
#         y_range.append(funct(x))
#     plt.plot(x_range,y_range,cl)
#     return
#
# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))
#
# rs=1.0
# r=np.linspace(-rs*np.pi,rs*np.pi,80)
# # v=lambda t: 15*np.sin(t)*np.power(np.cos(t),3)*np.exp(1/(t-19))
# v = lambda x: sigmoid(x)
# #v=lambda t: 15*np.sin(t**3)*np.power(np.cos(t**2),3)*np.exp(1/(t-19))
# PUv_coeffs = compute_PUv_coeffs(v, polynomial_degree+1)
# graph(lambda t: PUv(t,PUv_coeffs),r,cl='r-')
# graph(v,r,cl='b--')
# plt.axis('equal')
# plt.show()