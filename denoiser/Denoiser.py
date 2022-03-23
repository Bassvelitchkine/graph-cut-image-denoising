import numpy as np
import networkx as nx

class ImageDenoiser():
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
        self.reconstructed_image = np.copy(noisy_image) 


    def data_fitting(self):
        if self.Data_fitting_type == "L2":
            return np.sum((self.reconstructed_image - self.noisy_image)**2)
        else:
            return np.sum(np.abs(self.reconstructed_image-self.noisy_image))
    
    def psi(self, xi, xj):
        return np.abs(int(xi)-int(xj))   #Anisantropic TV for now. Transform to int beacause of overflow of uint8

    
    def energy(self):
        regularization = 0
        for edge in list(self.G.edges()):
            (x1, y1), (x2, y2) = edge
            ui = self.reconstructed_image[x1,y1]
            uj = self.reconstructed_image[x2,y2]
            regularization += self.psi(ui, uj)
        fit_to_data = self.data_fitting()
        return regularization + fit_to_data



