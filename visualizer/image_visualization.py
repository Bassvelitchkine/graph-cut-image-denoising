import matplotlib.pyplot as plt

def print_grayscale(img):
    '''
    A function to print a grayscale image with ```matplotlib.pyplot```.
    '''
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def print_rgb(img):
    '''
    A function to print a rgb image with ```matplotlib.pyplot```.
    '''
    plt.imshow(img)
    plt.show()
