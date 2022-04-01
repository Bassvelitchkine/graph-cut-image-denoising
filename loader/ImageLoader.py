import numpy as np
import networkx as nx

from skimage.io import imread
from skimage.transform import rescale

class ImageLoader():
    '''
    A class to load images from a data set, build a noisy version of this image
    and its 4-connected graph. 
    '''
    def __init__(self, img="image0.jpg", noise="S&P", seed=894397569, rescale_factor=None, build_graph=False):
        assert noise in ["S&P", "gaussian", "poisson"], print("noise parameter must be one of 'S&P', 'gaussian' or 'poisson")
        path = f"./data/{img}"
        original_image = imread(path).astype(int)

        # Define attributes
        self.rng_ = np.random.default_rng(seed=seed)
        self.original_image_ = original_image


        if len(np.shape(original_image)) == 3:
            self.grayscale_image_ = self.__to_grayscale(original_image)
        else:
            self.grayscale_image_ = np.copy(original_image)

        if rescale_factor:
            self.grayscale_image_ = rescale(self.grayscale_image_, scale=rescale_factor, preserve_range=True).astype(int)
        self.noisy_image_ = self.__add_noise(self.grayscale_image_, noise)

        if build_graph:
            G = nx.DiGraph()
            G.add_edges_from(self.__build_edge_list())
            self.graph_ = G
        else:
            self.graph_ = None

    def __to_grayscale(self, rgb_image):
        '''
        A function that loads a RGB image as a grayscale numpy array.

        Params:
        -------
            - rgb_image: np.array as returned by ```skimage.io.imread```
                The rgb image to turn into grayscale
        
        Outputs:
        --------
            - gray_img: np.array with int values ranging from 0 to 255
                The input image turned into grayscale.
        '''
        gray_img = rgb_image @ np.array([0.2989, 0.5870, 0.1140]).T
        return gray_img.astype(int)

    def __add_noise(self, img, noise_type):
        '''
        A function that adds noise to an image, whether its:
        - Salt and Pepper (S&P)
        - Poisson
        - Gaussian
        noise.

        Params:
        ------
            - img: np.array
                The image to add noise to.
            - noise_type: str
                Either "S&P", "poisson" or "gaussian". The type of noise to subject the image to.
        
        Outputs:
        -------
            - result: np.array
                The noisy image.
        '''
        height, width = img.shape
        result = img.copy()

        if noise_type == "S&P":
            for line in range(height):
                for column in range(width):
                    sample = self.rng_.random()
                    if sample < 0.05:
                        result[line, column] = 255 if self.rng_.random() < 0.5 else 0
            
        elif noise_type == "gaussian":
            for line in range(height):
                for column in range(width):
                    noise = self.rng_.random() * 50
                    new_pix_val = result[line, column] + noise
                    result[line, column] = np.clip(new_pix_val, 0, 255)

        elif noise_type == "poisson":
            for line in range(height):
                for column in range(width):
                    noise = self.rng_.poisson(lam=10) * (-1) ** (line * column)
                    new_pix_val = result[line, column] + noise
                    result[line, column] = np.clip(new_pix_val, 0, 255)

        return result

    def __build_edge_list(self):
        '''
        A private method to build an list of edges from an image. The structure of the graph is 4-connected.
        Therefore, for any pixel (i,j) that's not on the border of the picture, (i,j) is turned into a
        node, like every other pixel, and has common edges with:
        - (i+1, j)
        - (i, j+1)
        - (i-1, j)
        - (i, j-1)
        Hence, the '4-connected' graph.

        Params:
        ------
            - None
        
        Outputs:
        --------
            - edge_list: list
                The list of every edge in the graph to build.
        '''
        edge_list = []
        height, width = self.grayscale_image_.shape

        # We run through every pixel except those on the right-hand side border and bottom border
        for line in range(height - 1):
            for column in range(width - 1):
                ref_node = (line, column)
                right_node = (line, column+1)
                node_below = (line+1, column)
                edge_list += [(ref_node, right_node), (ref_node, node_below), (right_node, ref_node), (node_below, ref_node)]
        
        # We handle the bottom border
        for column in range(width - 1):
            ref_node = (height-1, column)
            right_node = (height-1, column+1)
            edge_list += [(ref_node, right_node), (right_node, ref_node)]
        
        # We handle the right-hand side border
        for line in range(height - 1):
            ref_node = (line, width-1)
            node_below = (line+1, width-1)
            edge_list += [(ref_node, node_below), (node_below, ref_node)]

        return edge_list

    def grayscale(self):
        '''
        Returns the grayscale image
        '''
        return self.grayscale_image_

    def image(self):
        '''
        Returns the original image
        '''
        return self.original_image_

    def noisy_image(self):
        '''
        Returns the noisy image
        '''
        return self.noisy_image_

    def graph(self):
        '''
        Builds and returns the graph generated from the image.

        Params:
        ------
            - None

        Outputs:
        --------
            - self.graph_: ```nx.Graph```
                The newly built 4-connected graph.
        '''
        if not(self.graph_):
            print("The graph had not been built before. Wait until its created...")
            G = nx.DiGraph()
            G.add_edges_from(self.__build_edge_list())
            self.graph_ = G
        return self.graph_
    