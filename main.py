from loader.ImageLoader import ImageLoader
from visualizer.image_visualization import print_grayscale, print_rgb

loader = ImageLoader("image0.jpg", rescale_factor=0.5)

# Print the original image
print_rgb(loader.image())

# Print the grayscale image
print_grayscale(loader.grayscale())

# Print the noisy image
print_grayscale(loader.noisy_image())

# # Print a few edges from the constructed graph
graph = loader.graph()
print(list(graph.edges.data())[:5])