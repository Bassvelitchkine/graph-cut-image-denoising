from matplotlib import image
import numpy as np
import networkx as nx

class ImageLoader():
    '''
    A class to load images from our data set
    '''
    def __init__(self, img="image0.jpg", noise="S&P", seed=0):
        assert noise in ["S&P", "gaussian", "poisson"], print("noise parameter must be one of 'S&P', 'gaussian' or 'poisson")
        path = f"./data/{img}"
        original_image = self.load_(path)

        # Define attributes
        self.seed_ = seed
        self.original_image_ = original_image
        self.grayscale_image_ = self.to_grayscale_(original_image)
        self.noisy_image_ = self.add_noise_(self.grayscale_image_, noise)

        # Build grap
        G = nx.Graph()
        G.add_weighted_edges_from(self.build_edge_list_())
        self.graph_ = G

    def load_(self, path):
        '''
        A function to load a specific image from our data set
        '''
        return np.array(image.imread(path))

    def to_grayscale_(self, rgb_image):
        '''
        A function that loads a RGB image as a grayscale numpy array
        '''
        gray_img = rgb_image @ np.array([0.2989, 0.5870, 0.1140]).T
        return gray_img.astype(np.uint8)

    def add_noise_(self, img, noise_type):
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

    def compute_edge_weight_(self, pixel1, pixel2):
        '''
        A function that computes the weight of an edge from pixel1 to pixel2
        '''
        val1, val2 = self.noisy_image_[pixel1], self.noisy_image_[pixel2]
        if val1 != val2:
            return 1/((float(val1) - float(val2)) ** 2)
        else:
            return 2.

    def build_edge_list_(self):
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
                    edge_weight = self.compute_edge_weight_((line, column), adj_pix)
                    edge_list.append((node, self.pixel_to_node_(adj_pix), edge_weight))
        
        # We handle the bottom border
        for column in range(width - 1):
            node = self.pixel_to_node_((height - 1, column))
            adj_node = self.pixel_to_node_((height- 1, column + 1))
            edge_weight = self.compute_edge_weight_((height - 1, column), (height - 1, column + 1))
            edge_list.append((node, adj_node, edge_weight))
        
        # We handle the right-hand side border
        for line in range(height - 1):
            node = self.pixel_to_node_((line, width - 1))
            adj_node = self.pixel_to_node_((line + 1, width - 1))
            edge_weight = self.compute_edge_weight_((line, width - 1), (line + 1, width - 1))
            edge_list.append((node, adj_node, edge_weight))
        
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
    