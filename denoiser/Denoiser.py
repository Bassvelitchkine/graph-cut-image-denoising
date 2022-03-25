import numpy as np
import networkx as nx
from tqdm import tqdm

class ImageDenoiser():
    '''
    A class to load denoise images using graph cuts
    '''
    def __init__(
        self,
        G,
        noisy_image,
        data_fitting_type="L2",
        regularization_type="Anisotropic TV",
        regularization_weight=1e-2,
        seed=5898624765):

        assert regularization_type in ["Anisotropic TV", "Isotropic TV", "Isotropic", "Huber", "Weighted Isotropic", "Weighted Iso. TV", "Anisotr.", "Anisotr. non-quadr."]
        assert data_fitting_type in ["L2", "L1"]

        self.G = G
        self.data_fitting_type = data_fitting_type
        self.regularization_type = regularization_type
        self.noisy_image = noisy_image

        self.rng_ = np.random.default_rng(seed=seed)

        ## Reconstructed image: We can try to experiment with different initialization
        # self.reconstructed_image = np.random.randint(0,256, size=np.shape(noisy_image))
        self.reconstructed_image = np.copy(noisy_image)

        self.regularization_weight = regularization_weight


    def data_fitting(self):
        if self.data_fitting_type == "L2":
            return np.sum((self.reconstructed_image - self.noisy_image)**2)
        else:
            return np.sum(np.abs(self.reconstructed_image-self.noisy_image))
    
    def psi(self, xi, xj):
        return np.abs(int(xi)-int(xj))   #Anisantropic TV for now. 
        #Transform to int beacause of overflow of uint8

    def unary_cost(self, x, y, value):
        if self.data_fitting_type == "L2":
            return (self.reconstructed_image[x,y] - value)**2
        else:
            return np.abs(self.reconstructed_image[x,y] - value)

    def energy(self, image):
        regularization = 0
        for edge in self.G.edges():
            (x1, y1), (x2, y2) = edge
            ui = image[x1,y1]
            uj = image[x2,y2]
            regularization += self.psi(ui, uj)
        fit_to_data = self.data_fitting()
        return self.regularization_weight * regularization + fit_to_data

    def create_alpha_beta_graph(self, alpha, beta):
        G_alpha_beta = nx.DiGraph()
        ebunch_to_add = []
        for x,y in self.G.nodes():
            if self.reconstructed_image[x,y] in [alpha, beta]:
                ebunch_to_add.append( (alpha, (x,y), {"capacity": self.unary_cost(x,y, alpha)}) )  #Alpha source
                ebunch_to_add.append( ((x,y), beta, {"capacity": self.unary_cost(x,y, beta)}) )    #Beta sink
        for edge in self.G.edges():
            (x1, y1), (x2, y2) = edge
            if (self.reconstructed_image[x1,y1] in [alpha, beta]) and (self.reconstructed_image[x2,y2] in [alpha, beta]):
                u1 = self.reconstructed_image[x1,y1]
                u2 = self.reconstructed_image[x2,y2]
                ebunch_to_add.append( ((x1,y1), (x2,y2), {"capacity": self.regularization_weight * self.psi(u1,u2)}) )
        G_alpha_beta.add_edges_from(ebunch_to_add)
        return G_alpha_beta
    
    def alpha_beta_swap(self, max_iter=100):
        for i in tqdm(range(max_iter)):
            try_image = np.copy(self.reconstructed_image)
            alpha = np.random.randint(0,256)
            beta = np.random.randint(0,256)
            if alpha == beta:
                continue
            G_alpha_beta = self.create_alpha_beta_graph(alpha, beta)
            cut_value, partition = nx.minimum_cut(G_alpha_beta, alpha, beta)
            alpha_partition, beta_partition = partition
            for edge in alpha_partition:
                if type(edge) is tuple:
                    (x,y) = edge
                    try_image[x,y] = alpha
            for edge in beta_partition:
                if type(edge) is tuple:
                    (x,y) = edge
                    try_image[x,y] = beta
            if self.energy(try_image) < self.energy(self.reconstructed_image):
                self.reconstructed_image = np.copy(try_image)
                print(self.energy(try_image))

        return self.energy(self.reconstructed_image)

