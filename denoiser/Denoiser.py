import numpy as np
import networkx as nx
from tqdm.notebook import tqdm

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
        Initialize object.

        Params:
        -------
            - G: ```networkx.Graph```
                The graph associated to the image to denoise
            - noisy_image: np.array
                The noisy image to denoise
            - data_fitting_type: str (default 'L2')
                Either 'L1' or 'L2' tells what loss to use when computing the energy of an image
            - regularization_type: str (default 'Anisotropic TV')
                Either "Anisotropic TV", "Isotropic TV", "Isotropic", "Huber", "Weighted Isotropic",
                "Weighted Iso. TV", "Anisotr." or "Anisotr. non-quadr." HOWEVER, at the moment,
                whatever the parameter, it's the Anisotropic TV that'll be computed.
            - regularization_weight: float (default 1e-2)
                A float telling how much the regularization should weight in the energy calculations
            - seed: int (default 5898624765)
                For reproducibility, to give as parameter to ```numpy.random.random_rng```.
        '''
        assert regularization_type in ["Anisotropic TV", "Isotropic TV", "Isotropic", "Huber", "Weighted Isotropic", "Weighted Iso. TV", "Anisotr.", "Anisotr. non-quadr."]
        assert data_fitting_type in ["L2", "L1"]

        self.G = G
        self.data_fitting_type = data_fitting_type
        self.regularization_type = regularization_type
        self.noisy_image = noisy_image

        self.rng_ = np.random.default_rng(seed=seed)

        self.reconstructed_image = np.copy(noisy_image)
        self.regularization_weight = regularization_weight


    def __data_fitting(self, image):
        '''
        A private method to compute the distance of the reconstructed image to
        the noisy image.

        Params:
        -------
            - image: np.array
                The image whose distance to compute, to the noisy image.
        
        Outputs:
        --------
            - dist: float
                The L1 distance between the input image and the original noisy image.
        '''
        if self.data_fitting_type == "L2":
            return np.sum((image - self.noisy_image) ** 2)
        else:
            return np.sum(np.abs(image - self.noisy_image))
    
    def __psi(self, xi, xj):
        '''
        A method to compute the L1 distance between to values

        It's the Anisantropic TV for now.

        Params:
        -------
            - xi: int
                Value of a first pixel/node
            - xj: int
                Value of a second pixel/node
        
        Outputs:
        --------
            - dist: float
                The absolute value between xi and xj
        '''
        return np.abs(xi - xj)

    def __unary_cost(self, x, y, value):
        '''
        A private method to compute the distance between
        a single pixel of the reconstructed image and a value.

        Params:
        -------
            - x: int
                The position of the pixel in the reconstructed image on the vertical axis
            - y: int
                The position of the pixel in the reconstructed image on the horizontal axis
            - value: float
                The float to compute the distance of the pixel to.
        
        Outputs:
        --------
            - distance: float
                Either the 'L1' or 'L2' distance depending on the ```self.data_fitting_type```
                value.
        '''
        if self.data_fitting_type == "L2":
            return (self.reconstructed_image[x, y] - value) ** 2
        else:
            return np.abs(self.reconstructed_image[x, y] - value)

    def energy(self, image):
        '''
        A method to compute the energy of a whole image. The energy
        is the value that we're trying to minimize in the process of
        denoising.

        Params:
        -------
            - image: np.array
                An image
        
        Outputs:
        --------
            - energy: float
                The energy of the image given as input
        '''
        regularization = 0
        for edge in self.G.edges():
            (x1, y1), (x2, y2) = edge
            ui = image[x1, y1]
            uj = image[x2, y2]
            regularization += self.__psi(ui, uj)
        fit_to_data = self.__data_fitting(image)
        return self.regularization_weight * regularization + fit_to_data

    def __create_alpha_beta_graph(self, alpha, beta):
        '''
        A subgraph has to be recreated at each step of the alpha-beta swap algorithm.
        This private method recreates the subgraph on which to perform alpha-beta swap.

        Params:
        -------
            - alpha: int
                The alpha value in the alpha-beta swap algorithm
            - beta: int
                The beta value in the alpha-beta swap algorithm
        
        Outputs:
        --------
            - alpha_beta_graph: ```networkx.DiGraph```
                The directed graph derived from the original graph on which
                to perform a step of alpha-beta swap.
        '''
        alpha_beta_graph = nx.DiGraph()
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
                ebunch_to_add.append( ((x1, y1), (x2, y2), {"capacity": self.regularization_weight * self.__psi(alpha, beta)}) )
        alpha_beta_graph.add_edges_from(ebunch_to_add)
        return alpha_beta_graph
    
    def alpha_beta_swap(self, max_iter=100, verbose=True):
        '''
        A public method to perform the alpha-beta swap algorithm on the noisy image,
        for as many steps as needed.

        Params:
        ------
            - max_iter: int (default 100)
                The maximum number of apha-beta swaps to perform
            - verbose: bool (default True)
                Whether to print information to the user in the middle of the process or not
        
        Outputs:
        --------
            - energy_history: list
                A list that keeps track of the energy value evolution after each alpha-beta swap
                step.
            - steps: list
                A list that encompasses steps at which the reconstructed image was kept instead of
                that of the previous step.
        '''
        old_energy = self.energy(self.reconstructed_image)
        energy_history = [old_energy]
        steps = [0]
        if verbose: print(f"Start energy : {old_energy}")
        for step in tqdm(range(max_iter), display=verbose):
            try_image = np.copy(self.reconstructed_image)
            alpha, beta = self.rng_.integers(256, dtype=int), self.rng_.integers(256, dtype=int)
            if alpha == beta or alpha not in self.reconstructed_image or beta not in self.reconstructed_image:
                continue
            alpha_beta_graph = self.__create_alpha_beta_graph(alpha, beta)
            _, partition = nx.minimum_cut(alpha_beta_graph, alpha, beta)
            alpha_partition, beta_partition = partition
            for edge in alpha_partition:
                if isinstance(edge, tuple):
                    (x, y) = edge
                    try_image[x, y] = alpha
            for edge in beta_partition:
                if isinstance(edge, tuple):
                    (x, y) = edge
                    try_image[x, y] = beta
            new_energy = self.energy(try_image)
            if new_energy < old_energy:
                energy_history.append(new_energy)
                steps.append(step + 1)
                old_energy = new_energy
                self.reconstructed_image = np.copy(try_image)
        if verbose: print(f"End energy : {self.energy(self.reconstructed_image)}")
        return energy_history, steps
