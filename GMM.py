from dis import dis
from turtle import distance
from defusedxml import DTDForbidden
from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import diff
from sklearn.mixture import GaussianMixture
from matplotlib import colors
from soupsieve import closest

def generate_clusters(centers, nb, dim):
    r_vectors = np.random.rand(nb * centers.shape[0], dim)
    n_r_vectors = r_vectors  / np.linalg.norm(r_vectors, axis=1)[:,np.newaxis]
    return centers + n_r_vectors.reshape((nb,centers.shape[0],dim))


def generate_centers(dim, distance):
    positive = np.identity(dim)
    negative = -positive
    points = np.concatenate((positive, negative))
    return distance * points

def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])

class MGauss:
    def __init__(self, mu,sigma):
        self.mu = mu
        self.sigma = sigma
        self.inv_sigma = np.linalg.inv(sigma)
        self.k = mu.shape[0]
        self.sigmaDet = np.linalg.det(sigma)

    def p(self, x):
       exponential =  np.exp( -0.5 *( ( x - self.mu) @ self.inv_sigma @ (x - self.mu)) )
       other = np.sqrt( (2* np.pi)**self.k * self.sigmaDet )
       return exponential / other

    
#===============================================================================================    
    
def closest(p, pts):
    distances = np.linalg.norm(p-pts, axis=1)
    return np.argsort(distances)[0]
        
   
   
def initiatePrior(initialClusters, priors, points):
    for ind, p in enumerate(points):
        cluster = closest(p, initialClusters) 
        priors[cluster ][ind] = 1
    return priors

def mean(values):
    return np.mean(values, axis=0)

def covariance(values):
    mu = mean(values)
    return (values - mu) @ (values - mu).T / len(mu)

def log_likelihood(prior, distributions, gammas):
    return 0

def newMu(priors, pts):
    Nk = np.sum(priors, axis=0)
    
    mus = priors @ pts / Nk
    return mus, Nk

def newSigmas(priors, pts, mus):
    difference = pts - mus
    sigmas = np.zeros((nb_clusters, dim, dim))
    for k,cluster in enumerate(sigmas):
        for n,point in enumerate(difference):
            cluster += priors[k,n] * point @ point.T 
    return sigmas


# def E():

def M(priors, pts ):
    mus, Nk = newMu(priors, pts)
    sigmas = newSigmas(priors, pts, mus) / Nk
    pis = Nk/ len(pts)
    return mus, sigmas, pis

# =====================================================================================
# =========================== UTILS ===================================================

def generate_pts(dim, pts_per_cluster):
    centers = generate_centers(dim, pts_per_cluster)
    res = generate_clusters(centers, pts_per_cluster, dim)
    pts = combine_dims(res,0,2)
    return pts
nb_clusters = 4

def plotRes(centers, whichCluster):
    print(centers)
    print(whichCluster.shape)
    cmap = colors.ListedColormap(['k','b','y','g','r'])


    plt.scatter(pts[:,0],pts[:,1], c=whichCluster, cmap=cmap)
    # plt.scatter(centers[:,0], centers[:,1],  s=10)
    plt.show()

# =====================================================================================
    

dim = 2
pts_per_cluster = 5
pts = generate_pts(dim, pts_per_cluster)

gmm = GaussianMixture(nb_clusters)
ogmm = gmm.fit_predict(pts)
centers = np.empty(shape=(gmm.n_components, dim))

mu = np.array([0,0])
sigma = np.array( [ [1,0], [0,1]])
x = np.array([0,0])

print("gauss ", MGauss(mu, sigma).p(x) )

initialCenters = np.random.rand(nb_clusters, dim)


gammas = np.zeros((nb_clusters, len(pts) ))
mus = np.zeros((nb_clusters, dim))
sigmas = np.zeros((nb_clusters, dim, dim))
priors = np.zeros((nb_clusters, len(pts) ))

gammas = initiatePrior(initialCenters, priors, pts)
print(gammas)



print(M(gammas, pts))


