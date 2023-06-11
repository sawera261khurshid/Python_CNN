import numpy as np
import itertools


class MaxPool2:
  def __init__(self):
    self.weights = None
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output
 

  def backward(self, d_L_d_out):
    d_L_d_input = np.zeros_like(self.last_input)

    # Pooling operation
    for i, j, f in itertools.product(range(self.last_input.shape[0] // 2),
                                     range(self.last_input.shape[1] // 2),
                                     range(self.last_input.shape[2])):
        i *= 2
        j *= 2
        
        im_region = self.last_input[i:i + 2, j:j + 2, f].reshape(2, 2)

        # Find the indices of the max value in the input region
        max_indices = np.unravel_index(np.argmax(im_region), (2, 2))

        # Compute the gradient
        d_L_d_input[i + max_indices[0], j + max_indices[1], f] = d_L_d_out[i // 2, j // 2, f]

    return d_L_d_input
#   def backward(self, d_L_d_out):
#     d_L_d_input = np.zeros(self.input.shape)
#     for i in range(0, d_L_d_out.shape[0]):
#       for j in range(0, d_L_d_out.shape[1]):
#         for f in range(d_L_d_out.shape[2]):
#           patch = self.input[i*2:i*2+2, j*2:j*2+2, f]
#           max_val = np.max(patch)
#           mask = patch == max_val
#           d_L_d_input[i*2:i*2+2, j*2:j*2+2, f] = mask * d_L_d_out[i, j, f]
#     return d_L_d_input 
        
#   def backward(self, d_L_d_out):
#     '''
#     Performs a backward pass of the maxpool layer.
#     Returns the loss gradient for this layer's inputs.
#     - d_L_d_out is the loss gradient for this layer's outputs.
#     '''
#     d_L_d_input = np.zeros(self.last_input.shape)

#     for im_region, i, j in self.iterate_regions(self.last_input):
#       h, w, f = im_region.shape
#       amax = np.amax(im_region, axis=(0, 1))

#       for i2 in range(h):
#         for j2 in range(w):
#           for f2 in range(f):
#             # If this pixel was the max value, copy the gradient to it.
#             if im_region[i2, j2, f2] == amax[f2]:
#               d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = np.max(d_L_d_out[i, j, f2])

# #              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

#     return d_L_d_input






class MaxPool2_3d:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, c, _ = image.shape
    new_h = h // 2
    new_w = w // 2
    new_c = c

    for i in range(new_h):
      for j in range(new_w):
        for k in range(new_c):
          im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2), k]
          yield im_region, i, j, k

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, c, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j, k in self.iterate_regions(input):
      output[i, j, k] = np.amax(im_region, axis=(0, 1, 2))

    return output

  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j, k in self.iterate_regions(self.last_input):
      h, w, c, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1, 2))

      for i2 in range(h):
        for j2 in range(w):
          for k2 in range(c):
            for f2 in range(f):
              # If this pixel was the max value, copy the gradient to it.
              if im_region[i2, j2, k2, f2] == amax[f2]:
                d_L_d_input[i * 2 + i2, j * 2 + j2, k2, f2] = d_L_d_out[i, j, k, f2]

    return d_L_d_input
