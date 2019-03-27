import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def convolve(image, kernel):

    '''
    
    Description: computes the convolution of the image with the convolution 
    filter (kernel).

    Inputs:
        -'image' (numpy.ndarray) = a 2-D array representing a grayscaled image.
        -'kernel' (numpy.ndarray) = a 2-D array for the convolution filter.

    Output:
        -'conv_image' (numpy.ndarray) = the convolved image.

    '''

    # Get the image and kernel shapes:
    h_img, w_img = image.shape
    h_ker, w_ker = kernel.shape
    
    # Initialize the convolved image to zero:
    conv_image = np.zeros((h_img, w_img))

    # Pad the image by extending the boundary values.
    # (Don't pad with zeros so the padding frame isnot detected as an edge.):
    padded_width = ((h_ker // 2, h_ker // 2), (w_ker // 2, w_ker // 2))
    padded_image = np.pad(image, padded_width, mode = 'edge')
    
    # Flip the convolution kernel so convolution is done by element-wise 
    kernel = np.fliplr(np.flipud(kernel))

    # Iterate over patches of the padded image to compute the convolution product:
    for i in range(h_img):
        for j in range(w_img):
            mult = np.multiply(padded_image[i:h_ker + i, j:w_ker + j], kernel)
            conv_image[i, j] = np.sum(mult)

    return conv_image

def gaussian_kernel(window_size, sigma):

    '''

    Description: computes a square matrix representation of the Gaussian kernel
    with mean zero and standard deviation 'sigma'.

    Inputs:
        -'window_size' (int) = an even positive integer giving the shape of the
        square output matrix.
        -'sigma' (float) = the standard deviation of the Gaussian.

    Output:
        -'kernel' (numpy.ndarray) = the 2-D matrix representation of the Gaussian
        kernel with mean zero and standard deviation 'sigma'.

    '''

    assert sigma > 0
    assert window_size > 0
    assert window_size % 2 == 1
    
    # Compute the matrix representation of the 'window_size' x 'window_size'
    # matrix representation of the Gaussian kernel:
    k = int((window_size - 1) / 2)
    kernel = np.array([[(i - k) ** 2 + (j - k) ** 2 for i in range(2 * k + 1)] \
                                                    for j in range(2 * k + 1)])
    kernel = np.exp(-(kernel / (2 * sigma ** 2)))
    kernel = (1 / (2 * np.pi * sigma ** 2)) * kernel

    return kernel

def get_nearest_neighbors(y, x, height, width):

    '''

    Description: finds all nearest neighbors (left, right, up, down, diagonals)
    of a point (x, y) within a (0 to 'height' - 1) x (0 to 'width' - 1) square.

    Inputs:
        -'y' (int) = height (ranges from 0 to 'height' - 1, 0 is the top of the square).
        -'x' (int) = width  (ranges from 0 to 'width' - 1).
        -'height' (int) = the height of the square.
        -'width'  (int) = the width of the square.

    Output:
        -'neighbor_list' (list) = a list of all nearest neighbors of '(x, y)'.

    '''

    assert isinstance(y, int)
    assert isinstance(x, int)

    assert isinstance(height, int)
    assert isinstance(width,  int)

    assert height > 0
    assert width > 0

    assert y in range(height)
    assert x in range(width)

    neighbor_list = []

    # Iterate over all nearest neigbors of (i, j):
    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < height and j >= 0 and j < width:

                # A point cannot be one of its nearest neighbors:
                if (i == y and j == x):
                    continue

                neighbor_list +=[(i, j)]

    return neighbor_list

def show_image(image, title = 'No Title'):

    '''
    
    Description: displays 'image' inline within a Jupyter notebook, with title 'title'.

    Inputs:
        -'image' (numpy.ndarray) = a 2-D array representing the grayscaled image.
        -'title' (str) = the title for the image.

    '''

    plt.imshow(image, cmap = 'gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

class CannyEdgeDetector():

    '''

    Description: implementation of the Canny edge detection algorithm.
    More can be found here: https://en.wikipedia.org/wiki/Canny_edge_detector.

    '''

    def __init__(self,
                 sigma = 1.4,
                 kernel_size = 5, 
                 upper_threshold = 0.03, 
                 lower_threshold = 0.02):

        '''
        
        Description: initializer for the class instance of 'CannyEdgeDetector'.

        Inputs:
            -'sigma' (float) = the standard deviation of the Gaussian kernel used
            for smoothing.
            -'kernel_size' (int) = the window size for the Gaussian kernel used
            for smoothing.
            -'upper_threshold' (float) = upper threshold used for double-thresholding.
            -'lower_threshold' (float) = lower threshold used for double-thresholding.

        '''

        assert sigma > 0
        assert kernel_size > 0
        assert upper_threshold > 0
        assert lower_threshold > 0

        assert isinstance(kernel_size, int)

        self.sigma = sigma
        self.kernel_size = kernel_size

        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.gaussian_kernel = gaussian_kernel(kernel_size, sigma)

    def load_image_to_np_array(self, image_name):

        '''
        
        Description: opens an image, converts it to grayscale, and converts
        the result to a 2-D numpy array.

        Inputs:
            -'image_name' (str) = the file name of the image to open.

        '''

        # Load the image withot reformatting:
        image = Image.open(image_name) # Open the image as a PIL object.
        self.original_image = image

        # Reformat the image:
        image = image.convert('L')     # Convert it to grayscale.
        image = np.asarray(image)      # Convert it to a numpy array.
        image = np.true_divide(image, np.amax(image)) # Scale the image.

        # Check that the image is not smaller than the Gaussian kernel:
        assert self.kernel_size <= min(image.shape)

        # Write to a class attribute:
        self.image = image

    def smooth_image(self):

        '''
        
        Description: produces a smoothed image of 'self.image' by convolving it
        with 'self.gaussian_kernel'.

        '''

        assert hasattr(self, 'image')

        # Write to a class attribute:
        self.smoothed_image = convolve(self.image, self.gaussian_kernel)

    def compute_gradients(self):

        '''
        
        Description: computes the magnitudes and directions of all gradients
        of 'self.smoothed_image'. Directions are in the range [0, 360).

        '''

        assert hasattr(self, 'smoothed_image')

        # Kernels for differentiation with respect to the x and y directions:
        dx_kernel = np.array([[1/2, 0, -1/2]])
        dy_kernel = np.array([[1/2], [0], [-1/2]])

        # Compute the x and y derivatives of the smoothed image:
        self.image_dx = convolve(self.smoothed_image, dx_kernel)
        self.image_dy = convolve(self.smoothed_image, dy_kernel)

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            grad_magnitudes = np.sqrt(np.square(self.image_dx) + np.square(self.image_dy))
            grad_directions = np.arctan(np.true_divide(-self.image_dy, self.image_dx))
            grad_directions = np.nan_to_num(grad_directions)
            grad_directions += 2 * np.pi * np.greater(np.zeros(grad_directions.shape), grad_directions)
            grad_directions *= 180 / (np.pi)
            grad_directions[grad_directions == 360.0] = 0.0

        # Check that the shapes are correct:
        assert grad_magnitudes.shape == self.image.shape
        assert grad_directions.shape == self.image.shape

        # Write to class attributes:
        self.grad_magnitudes = grad_magnitudes
        self.grad_directions = grad_directions

    def apply_non_maximum_suppression(self):

        '''
        
        Description: apply non-maximum suppression of each gradient magnitude
        along the corresponding gradient direction (rounded to the nearest 45 degrees).

        '''

        assert hasattr(self, 'grad_magnitudes')
        assert hasattr(self, 'grad_directions')

        # Make copies of the grad magnitudes and directions:
        grad_magnitudes = np.copy(self.grad_magnitudes)
        grad_directions = np.copy(self.grad_directions)

        # Initialize the output image to zeros:
        height, width = grad_magnitudes.shape
        non_max_supp_image = np.zeros((height, width))

        # Pad the gradients:
        pad_width = ((1, 1), (1, 1))
        grads = np.pad(grad_magnitudes, pad_width, mode = 'constant')

        # Round the gradient direction to the nearest 45 degrees:
        theta = np.floor((grad_directions + 22.5) / 45) * 45
        theta[theta == 360.0] = 0.0

        # Check that each value of theta is a value allowed by the rounding:
        assert np.isin(theta, [45.0 * i for i in range(8)]).all()
        
        # Iterate through each pixel and use non-gradient suppression to decide
        # whether or not to include this pixel as a prospective edge:
        for i in range(height):
            for j in range(width):

                ip = i + 1
                jp = j + 1

                if theta[i, j] in [0.0, 180.0] \
                and grads[ip, jp] >= np.max([grads[ip, jp - 1], grads[ip, jp + 1]]):
                    non_max_supp_image[i, j] = grads[ip, jp]

                elif theta[i, j] in [90.0, 270.0] \
                and grads[ip, jp] >= np.max([grads[ip - 1, jp], grads[ip + 1, jp]]):
                    non_max_supp_image[i, j] = grads[ip, jp]

                elif theta[i, j] in [45.0, 225.0] \
                and grads[ip, jp] >= np.max([grads[ip + 1, jp - 1], grads[ip - 1, jp + 1]]):
                    non_max_supp_image[i, j] = grads[ip, jp]

                elif theta[i, j] in [135.0, 315.0] \
                and grads[ip, jp] >= np.max([grads[ip + 1, jp + 1], grads[ip - 1, jp - 1]]):
                    non_max_supp_image[i, j] = grads[ip, jp]

        # Check that the shape is correct:
        assert non_max_supp_image.shape == self.image.shape

        # Write to a class attribute:
        self.non_max_supp_image = non_max_supp_image

    def apply_double_thresholding(self):

        '''
        
        Description: keep only gradients in 'self.non_max_supp_image' that are
        large enough to live in '(self.lower_threshold, self.upper_threshold)'
        (so-called "light edges") or greater than 'self.upper_threshold' (so-called
        "heavy edges").

        '''

        assert hasattr(self, 'non_max_supp_image')

        # Make a copy of the image with non-maximum suppression:
        image = np.copy(self.non_max_supp_image)
        shape = image.shape

        # Keep only edges (boolean) that fall within the given threshold ranges:
        heavy_edges = np.greater(image, self.upper_threshold * np.ones(shape))
        light_edges = np.greater(image, self.lower_threshold * np.ones(shape)) \
                    & np.greater(self.upper_threshold * np.ones(shape), image)

        # Check that no site is the location of both a heavy edge and a light edge:
        assert not np.logical_and(heavy_edges, light_edges).any()

        # Check that the shapes are correct:
        assert heavy_edges.shape == self.image.shape
        assert light_edges.shape == self.image.shape

        # Write to class attributes:
        self.heavy_edges = heavy_edges
        self.light_edges = light_edges

    def compute_final_edges(self):

        '''
        
        Description: keep all edges of 'self.light_edges' that are nearest 
        neighbors of edges in 'self.heavy_edges', or are nearest neighbors
        of nearest neighbors of edges in 'self.heavy_edges', etc. The final
        result is the Canny edge detector of 'self.image'.

        '''

        assert hasattr(self, 'heavy_edges')
        assert hasattr(self, 'light_edges')

        # Make copies of the heavy edges and light edges:
        heavy_edges = np.copy(self.heavy_edges)
        light_edges = np.copy(self.light_edges)

        # Get the shape of 'heavy_edges' (also the shape of 'light_edges'):
        height, width = heavy_edges.shape

        # Initialize the final edges matrix to all zeros:
        edges = np.zeros((height, width), dtype = np.bool)

        # Get the indices of all heavy edges:
        indices_list = [(int(i), int(j)) for i, j in zip(*np.nonzero(heavy_edges))]

        # Include in the final 'edges' all light edges that are adjacent to heavy
        # edges, or adjacent to light edges that are adjacent to heavy edges, etc.
        while len(indices_list) > 0:

            # Keep this edge:
            i, j = indices_list.pop(0)
            edges[i, j] = True

            # Find all of this edge's neighboring light edges, and add them to
            # 'indices_list' so they are included in the final 'edges':
            neighbor_list = get_nearest_neighbors(i, j, height, width)
            for ip, jp in neighbor_list:
                if light_edges[ip, jp] == 1:
                    light_edges[ip, jp] = 0
                    indices_list += [(ip, jp)]

        self.edges = edges

    def get_canny_edges(self, image_name, verbose = False):

        '''
        
        Description: compute the Canny edge detection algorithm to 'self.image'.

        Inputs:
            -'image_name' (str) = the file name of the image to open.
            -'verbose' (bool) = option to print out intermediate images.

        '''

        # Load the image to a numpy array:
        self.load_image_to_np_array(image_name)
        if verbose: self.show_original_image()
        
        # Smooth the image:
        self.smooth_image()
        if verbose: self.show_smoothed_image()

        # Compute the gradient magnitudes and directions:
        self.compute_gradients()
        if verbose: self.show_grad_magnitudes()
        if verbose: self.show_grad_directions()

        # Apply non-maximum suppression to all gradients:
        self.apply_non_maximum_suppression()
        if verbose: self.show_non_max_supp_image()

        # Apply double-thresholding to all non-supressed gradients:
        self.apply_double_thresholding()
        if verbose: self.show_light_edges()
        if verbose: self.show_heavy_edges()

        # Link weak edges to adjacent strong edges, or otherwise drop them:
        self.compute_final_edges()
        if verbose: self.show_edges()

        return self.edges

    # Method for displaying the images generated in the various steps of the
    # Canny edge detection algorithm:

    def show_original_image(self, title = 'Original Image'):
        show_image(self.image, title)

    def show_smoothed_image(self,  title = 'Smoothed Image'):
        show_image(self.smoothed_image, title)

    def show_grad_magnitudes(self, title = 'Gradient Magnitudes'):
        show_image(self.grad_magnitudes, title)

    def show_grad_directions(self, title = 'Gradient Directions'):
        show_image(self.grad_directions, title)

    def show_non_max_supp_image(self, title = 'Maximum Gradients'):
        show_image(self.non_max_supp_image, title)

    def show_light_edges(self, title = 'Light Edges'):
        show_image(self.light_edges, title)

    def show_heavy_edges(self, title = 'Heavy Edges'):
        show_image(self.heavy_edges, title)

    def show_edges(self, title = 'Detected Edges'):
        show_image(self.edges, title)

    def show_image_and_edges(self, image_name):

        '''
        
        Description: a plotter showing the side-by-side comparison of
        an original image with its detected edges in a Jupyter notebook.

        Inputs:
            -'image_name' (str) = the file name of the image to open.

        '''

        self.get_canny_edges(image_name)

        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image)
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(self.edges, cmap = 'gray')
        plt.axis('off')
        plt.title('Detected Edges')

        plt.show()
