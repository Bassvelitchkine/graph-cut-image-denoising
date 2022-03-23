import numpy as np
import networkx as nx
from tqdm import tqdm

class ImageDenoiser():
    '''
    A class to load denoise images using graph cuts
    '''
    def __init__(self, G, noisy_image, Data_fitting_type="L2", Regularization_type="Anisotropic TV", regularization_weight=1e-2):
        assert Regularization_type in ["Anisotropic TV", "Isotropic TV", "Isotropic", "Huber", "Weighted Isotropic", "Weighted Iso. TV", "Anisotr.", "Anisotr. non-quadr."]
        assert Data_fitting_type in ["L2", "L1"]
        self.G = G
        self.Data_fitting_type = Data_fitting_type
        self.Regularization = Regularization_type
        self.noisy_image = noisy_image
        self.reconstructed_image = np.random.randint(0,256, size=np.shape(noisy_image))
        self.regularization_weight = regularization_weight


    def data_fitting(self):
        if self.Data_fitting_type == "L2":
            return np.sum((self.reconstructed_image - self.noisy_image)**2)
        else:
            return np.sum(np.abs(self.reconstructed_image-self.noisy_image))
    
    def psi(self, xi, xj):
        return np.abs(int(xi)-int(xj))   #Anisantropic TV for now. 
        #Transform to int beacause of overflow of uint8

    def unary_cost(self, x, y, value):
        if self.Data_fitting_type == "L2":
            return (self.reconstructed_image[x,y] - value)**2
        else:
            return np.abs(self.reconstructed_image[x,y] - value)

    def energy(self):
        regularization = 0
        for edge in self.G.edges():
            (x1, y1), (x2, y2) = edge
            ui = self.reconstructed_image[x1,y1]
            uj = self.reconstructed_image[x2,y2]
            regularization += self.psi(ui, uj)
        fit_to_data = self.data_fitting()
        return self.regularization_weight * regularization + fit_to_data

    def create_alpha_beta_graph(self, alpha, beta):
        G_alpha_beta = self.G.copy()
        nodes_to_remove = []
        ebunch_to_add = []
        attrs = {}
        for x,y in self.G.nodes():
            if self.reconstructed_image[x,y] not in [alpha, beta]:
                nodes_to_remove.append((x,y))
                ebunch_to_add.append( ((x,y), alpha, {"capacity": self.unary_cost(x,y, alpha)}) )
                ebunch_to_add.append( ((x,y), beta, {"capacity": self.unary_cost(x,y, beta)}) )
        G_alpha_beta.remove_nodes_from(nodes_to_remove)
        for edge in G_alpha_beta.edges():
            (x1, y1), (x2, y2) = edge
            ui = self.reconstructed_image[x1,y1]
            uj = self.reconstructed_image[x2,y2]
            attrs[edge] = {"capacity": self.regularization_weight*self.psi(ui,uj)}
        G_alpha_beta.add_edges_from(ebunch_to_add)
        nx.set_edge_attributes(G_alpha_beta, attrs)
        return G_alpha_beta
    
    def alpha_beta_swap(self, max_iter=1):
        for i in range(max_iter):
            alpha = np.random.randint(0,256)
            beta = np.random.randint(0,256)
            if alpha == beta:
                continue
            G_alpha_beta = self.create_alpha_beta_graph(alpha, beta)
            cut_value, partition = nx.minimum_cut(G_alpha_beta, alpha, beta)
            reachable, non_reachable = partition

            ## TODO 
            """
            Do the cut
            Fin the new labels of the remaining pixels on the graph
            Evaluate energy, if energy decreases associate the new labels to the reconstructed image
            loop
            Ressource https://julie-jiang.github.io/image-segmentation/  Maybe we need to to a pre processing about the flow
            """
            return G_alpha_beta

