import numpy as np
import networkx as nx

class ImageLoader():
    '''
    A class to load denoise images using graph cuts
    '''
    def __init__(self, G, noisy_image, Data_fitting_type="L2", Regularization_type="Anisotropic TV", alpha=1e-2):
        assert Regularization_type in ["Anisotropic TV", "Isotropic TV", "Isotropic", "Huber", "Weighted Isotropic", "Weighted Iso. TV", "Anisotr.", "Anisotr. non-quadr."]
        assert Data_fitting_type in ["L2", "L1"]
        self.G = G
        self.Data_fitting_type = Data_fitting_type
        self.Regularization = Regularization_type
        self.noisy_image = noisy_image


    def data_fitting(self, reconstructed_image, noisy_image):
        if self.Data_fitting_type == "L2":
            return np.sum((reconstructed_image - noisy_image)**2)
        else:
            return np.sum(np.abs(reconstructed_image-noisy_image))
                



