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
        '''
        Initialize object
        '''
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


    def __data_fitting(self):
        '''
        A method to compute the distance of the reconstructed image to
        the noisy image
        '''
        if self.data_fitting_type == "L2":
            return np.sum((self.reconstructed_image - self.noisy_image) ** 2)
        else:
            return np.sum(np.abs(self.reconstructed_image - self.noisy_image))
    
    def __psi(self, xi, xj):
        '''
        A method to compute the L1 distance between to values

        It's the Anisantropic TV for now.
        '''
        #Transform to int beacause of overflow of uint8
        return np.abs(int(xi) - int(xj))

    def __unary_cost(self, x, y, value):
        '''
        Distance between a single pixel of the reconstructed image and a value
        '''
        if self.data_fitting_type == "L2":
            return (self.reconstructed_image[x, y] - value) ** 2
        else:
            return np.abs(self.reconstructed_image[x, y] - value)

    def __energy(self, image):
        '''
        A method to compute the energy of a whole image
        as defined in papers
        '''
        regularization = 0
        for edge in self.G.edges():
            (x1, y1), (x2, y2) = edge
            ui = image[x1, y1]
            uj = image[x2, y2]
            regularization += self.__psi(ui, uj)
        fit_to_data = self.__data_fitting()
        return self.regularization_weight * regularization + fit_to_data

    def __create_alpha_beta_graph(self, alpha, beta):
        '''
        A subgraph has to be recreated at each step of the alpha-beta swap algorithm.
        This method recreates the graph.
        '''
        G_alpha_beta = nx.DiGraph()
        ebunch_to_add = []
        for x, y in self.G.nodes():
            if self.reconstructed_image[x, y] in [alpha, beta]:
                # Alpha source
                ebunch_to_add.append( (alpha, (x, y), {"capacity": self.__unary_cost(x, y, alpha)}) )
                # Beta sink
                ebunch_to_add.append( ((x, y), beta, {"capacity": self.__unary_cost(x, y, beta)}) )
        for edge in self.G.edges():
            (x1, y1), (x2, y2) = edge
            if (self.reconstructed_image[x1, y1] in [alpha, beta]) and (self.reconstructed_image[x2, y2] in [alpha, beta]):
                u1 = self.reconstructed_image[x1, y1]
                u2 = self.reconstructed_image[x2, y2]
                ebunch_to_add.append( ((x1, y1), (x2, y2), {"capacity": self.regularization_weight * self.__psi(u1, u2)}) )
        G_alpha_beta.add_edges_from(ebunch_to_add)
        return G_alpha_beta
    
    def alpha_beta_swap(self, max_iter=100):
        '''
        A public method to perform the alpha-beta swap on the noisy image
        '''
        for _ in tqdm(range(max_iter)):
            try_image = np.copy(self.reconstructed_image)
            alpha, beta = self.rng_.integers(256), self.rng_.integers(256)
            if alpha == beta:
                continue
            G_alpha_beta = self.__create_alpha_beta_graph(alpha, beta)
            _, partition = nx.minimum_cut(G_alpha_beta, alpha, beta)
            alpha_partition, beta_partition = partition
            for edge in alpha_partition:
                if isinstance(edge, tuple):
                    (x, y) = edge
                    try_image[x, y] = alpha
            for edge in beta_partition:
                if isinstance(edge, tuple):
                    (x, y) = edge
                    try_image[x, y] = beta
            if self.__energy(try_image) < self.__energy(self.reconstructed_image):
                self.reconstructed_image = np.copy(try_image)
                print(self.__energy(try_image))

        return self.__energy(self.reconstructed_image)
