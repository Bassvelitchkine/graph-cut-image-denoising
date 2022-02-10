import matplotlib.pyplot as plt

def print_grayscale(img):
    '''
    A function to print a grayscale image
    '''
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()
