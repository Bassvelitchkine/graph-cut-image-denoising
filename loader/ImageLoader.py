from skimage.io import imread
from skimage.transform import rescale
import numpy as np
import networkx as nx

class ImageLoader():
    '''
    A class to load images from our data set
    '''
    def __init__(self, img="image0.jpg", noise="S&P", seed=0, rescale_factor=None):
        assert noise in ["S&P", "gaussian", "poisson"], print("noise parameter must be one of 'S&P', 'gaussian' or 'poisson")
        path = f"./data/{img}"
        original_image = imread(path)

        # Define attributes
        self.seed_ = seed
        self.original_image_ = original_image

        if rescale_factor:
            original_image = rescale(original_image, scale=rescale_factor,channel_axis=-1, preserve_range=True)

        self.grayscale_image_ = self.__to_grayscale(original_image)
        self.noisy_image_ = self.__add_noise(self.grayscale_image_, noise)

        # Build grap
        G = nx.Graph()
        G.add_edges_from(self.__build_edge_list())
        self.graph_ = G

    def __to_grayscale(self, rgb_image):
        '''
        A function that loads a RGB image as a grayscale numpy array
        '''
        gray_img = rgb_image @ np.array([0.2989, 0.5870, 0.1140]).T
        return gray_img.astype(np.uint8)

    def __add_noise(self, img, noise_type):
        '''
        A function that adds noise to an image
        '''
        height, width = img.shape
        result = img.copy()
        np.random.seed(self.seed_)

        if noise_type == "S&P":
            for line in range(height):
                for column in range(width):
                    sample = np.random.random()
                    if sample < 0.05:
                        result[line, column] = 255 if np.random.sample() < 0.5 else 0
            
        elif noise_type == "gaussian":
            for line in range(height):
                for column in range(width):
                    noise = np.random.random() * 20
                    new_pix_val = result[line, column] + noise
                    result[line, column] = np.clip(new_pix_val, 0, 255)

        elif noise_type == "poisson":
            for line in range(height):
                for column in range(width):
                    noise = np.random.poisson(lam=10) * (-1) ** (line * column)
                    new_pix_val = result[line, column] + noise
                    result[line, column] = np.clip(new_pix_val, 0, 255)

        return result

    def pixel_to_node_(self, pixel):
        '''
        Converts a pixel to its node number. For a pixel (i, j) (line i, column j), we compute a unique number.
        '''
        _, width = self.grayscale_image_.shape
        return pixel[0] * width + pixel[1]
    
    def node_to_pixel_(self, node):
        '''
        Given a node number, the function computes the pixel position that corresponds to the node in the original image.
        '''
        _, width = self.grayscale_image_.shape
        column = node % width 
        line = node // width
        return line, column

    def __build_edge_list(self):
        '''
        A function that builds a graph from an image. We draw edges between each pixel and its 8 adjacent pixels.
        That means that when at pixel (i,j), we draw an edge between that pixel and (i, j+1), (i+1, j), (i+1, j+1)
        '''
        edge_list = []
        height, width = self.grayscale_image_.shape

        # We run through every pixel except those on the right-hand side border and bottom border
        for line in range(height - 1):
            for column in range(width - 1):
                node = self.pixel_to_node_((line, column))
                adjacent_pixels = [(line, column + 1), (line + 1, column), (line + 1, column + 1)]
                for adj_pix in adjacent_pixels:
                    edge_list.append((node, self.pixel_to_node_(adj_pix)))
        
        # We handle the bottom border
        for column in range(width - 1):
            node = self.pixel_to_node_((height - 1, column))
            adj_node = self.pixel_to_node_((height- 1, column + 1))
            edge_list.append((node, adj_node))
        
        # We handle the right-hand side border
        for line in range(height - 1):
            node = self.pixel_to_node_((line, width - 1))
            adj_node = self.pixel_to_node_((line + 1, width - 1))
            edge_list.append((node, adj_node))
        
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
        Returns the graph generated from the image
        '''
        return self.graph_
    