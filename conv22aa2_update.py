import numpy as np

'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    np.save('conv_filters',self.filters)

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
      
    self.last_output = output
    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None


  
  
class Conv3x3_padding:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    np.save('conv_filters',self.filters)

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)[:,:]
    h, w = image_padded.shape[:2]

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image_padded[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def iterate_regions_3d(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)[:,:,1:-1]
    h, w, c = image_padded.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image_padded[i:(i + 3), j:(j + 3), :]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w, c = input.shape
    output = np.zeros((h , w , self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
      
    self.last_output = output
    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''

    # d_L_d_weights = d_L_d_out * inputs
    # -------------------------------------------------------------
    d_L_d_filters = np.zeros(self.filters.shape)
    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region



    # d_L_d_input   = d_L_d_out * weights (convolve)
    # -------------------------------------------------------------
    d_L_d_input = np.zeros(self.last_input.shape)
    for im_region, i, j in self.iterate_regions_3d(d_L_d_out):
      for f in range(self.num_filters):
        d_L_d_input[i,j] += np.sum(  self.filters[f,:,:] * im_region[:,:,f]  )
    
    
    
    
    # Update filters
    self.filters -= learn_rate * d_L_d_filters
    
    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return d_L_d_input





class Conv3x3_1_to_n_padding:
  # A Convolution layer using 3x3 filters.

  def __init__(self, output=1, inputch=1, activation=None, dtype=np.float32):
    num_filters = output
    in_ch = inputch
    self.activation = activation
    self.dtype = dtype
    self.num_filters = num_filters
    self.in_ch = in_ch

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3).astype(dtype) / 9
    np.save('conv_filters_3d',self.filters)

  def initialize_weights(self):
    self.filters = np.random.randn(self.num_filters, 3, 3).astype(self.dtype) / 9

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)
    h, w = image_padded.shape[:2]
    for i in range(h - 2):
      for j in range(w - 2):
          im_region = image_padded[i:(i + 3), j:(j + 3)]
          yield im_region, i, j
    
  def forward(self, input):
    self.last_input = input
    h, w, in_ch = input.shape
    output = np.zeros((h, w , self.num_filters), dtype=self.dtype)

    for im_region, i, j in self.iterate_regions(input):
      for f in range(self.num_filters):
        output[i, j, f] = np.sum(im_region * self.filters[f], axis=(0, 1, 2))
        self.last_output = output
        if self.activation is not None:
          output = self.activation(output)
        return output

  def backward(self, d_L_d_out, learn_rate):
    d_L_d_filters = np.zeros(self.filters.shape, dtype=self.dtype)
    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[:, :, :, f] += np.sum(im_region * d_L_d_out[i, j, f], axis=(0, 1))
        d_L_d_input = np.zeros(self.last_input.shape, dtype=self.dtype)
        for im_region, i, j in self.iterate_regions(d_L_d_out):
          d_L_d_input[i:i + self.kernel_size, j:j + self.kernel_size] += np.sum(
            self.filters[:, :, :, :] * im_region[:, :, np.newaxis, np.newaxis], axis=(3, 4))
          self.filters -= learn_rate * d_L_d_filters
          self.biases -= learn_rate * np.sum(d_L_d_out, axis=(0, 1))
          return d_L_d_input
      
  def get_weights(self):
    return self.filters

  def set_weights(self, filters):
    self.filters = filters

#   def forward(self, input):
#     '''
#     Performs a forward pass of the conv layer using the given input.
#     Returns a 3d numpy array with dimensions (h, w, num_filters).
#     - input is a 2d numpy array
#     '''
#     self.last_input = input

#     h, w = input.shape
#     output = np.zeros((h, w , self.num_filters), dtype = self.dtype)
#     # output = np.zeros((h - 2, w - 2, self.num_filters))

#     for im_region, i, j in self.iterate_regions(input):
#       output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

#     self.last_output = output
#     if self.activation is not None:
#         output = self.activation(output)
#     return output

#   def backward(self, d_L_d_out, learn_rate):
#     '''
#     Performs a backward pass of the conv layer.
#     - d_L_d_out is the loss gradient for this layer's outputs.
#     - learn_rate is a float.
#     '''
#     d_L_d_filters = np.zeros(self.filters.shape, dtype = self.dtype)
#     d_L_d_input   = np.zeros(self.last_input.shape, dtype = self.dtype) 
    
#     if self.activation is not None:
#         d_L_d_out = self.activation.backward(self.last_output) * d_L_d_out

#     for im_region, i, j in self.iterate_regions(self.last_input):
#       for f in range(self.num_filters):
#         d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
    
#     for out_ch in range(self.num_filters):
#       d_L_d_input[:,:] += d_L_d_out[:,:, out_ch] * self.last_input[:,:]
        
        
#     # errors of previous layer = weights_of_this_layer-T * errors of this layer
    
#     # Update filters
#     self.filters -= learn_rate * d_L_d_filters
    
#     return d_L_d_input






class Conv3x3_n_to_n_padding:
  # A Convolution layer using 3x3 filters.

  def __init__(self, output=1, inputch=1, activation=None, dtype=np.float32):
    num_filters = output
    in_ch = inputch
    self.activation = activation
    self.dtype = dtype
    self.num_filters = num_filters
    self.in_ch = in_ch

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3, in_ch).astype(dtype) / 9
    np.save('conv_filters_3d',self.filters)
    
    # initilize weights
  def initialize_weights(self):
    self.filters = np.random.randn(self.num_filters, 3, 3, self.in_ch).astype(self.dtype) / 9

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    image_padded = np.pad(image,1,mode="constant", constant_values=0)[:,:,1:-1]
    h, w = image_padded.shape[:2]
    for i in range(h - 2):
      for j in range(w - 2):
          im_region = image_padded[i:(i + 3), j:(j + 3), :]
          yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w, c = input.shape
    output = np.zeros((h, w , self.num_filters), dtype = self.dtype)
    # output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      for filter in range(self.num_filters):
        output[i, j, filter] = np.sum(im_region * self.filters[filter,:,:,:], axis=(0, 1, 2))

    
    self.last_output = output
    if self.activation is not None:
        output = self.activation(output)
    return output
      
  def backward(self, d_L_d_out, learn_rate):
      d_L_d_filters = np.zeros(self.filters.shape, dtype=self.dtype)
      d_L_d_input = np.zeros(self.last_input.shape, dtype=self.dtype)

      for im_region, i, j in self.iterate_regions(self.last_input):
          for f in range(self.num_filters):
              d_L_d_filters[:, :, :, f] += np.sum(im_region * d_L_d_out[i, j], axis=(0, 1))
              d_L_d_input[i:i + self.kernel_size, j:j + self.kernel_size, :] += np.sum(self.filters[:, :, :, f] * d_L_d_out[i, j], axis=2)

      # Update weights and biases
      self.filters -= learn_rate * d_L_d_filters
      self.biases -= learn_rate * np.sum(d_L_d_out, axis=(0, 1))

      return d_L_d_input
    
  def get_weights(self):
    return self.filters
  def set_weights(self, filters):
    self.filters = filters







#def backward(self, d_L_d_out, learn_rate):
#     '''
#     Performs a backward pass of the conv layer.
#     - d_L_d_out is the loss gradient for this layer's outputs.
#     - learn_rate is a float.
#     '''
#     d_L_d_filters = np.zeros(self.filters.shape, dtype = self.dtype)
#     if self.activation is not None:
#         d_L_d_out = self.activation.backward(self.last_output) * d_L_d_out

#     for im_region, i, j in self.iterate_regions(self.last_input):
#       for f in range(self.num_filters):
#         d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
    
    
    
#     d_L_d_input   = np.zeros(self.last_input.shape, dtype = self.dtype) 
      
#     # Method 4
#     for im_region, i, j in self.iterate_regions(d_L_d_out):
#       for in_ch in range(d_L_d_input.shape[-1]):
#         # d_L_d_input[i,j,in_ch] += np.sum ( im_region[:,:,:] * np.transpose( self.filters[:,:,:,in_ch]) , axis=(0,1,2) )
# #         d_L_d_input[i,j,in_ch] += np.sum( np.matmul( im_region[:,:,:] , np.transpose( self.filters[:,:,:,in_ch] , axes=(2,0,1)) ) , axis=(0,1,2) )
#           d_L_d_input[i, j, in_ch] += np.sum(im_region * self.filters[:, :, :, in_ch], axis=(0, 1, 2))


#     # Update filters
#     self.filters -= learn_rate * d_L_d_filters

#     return d_L_d_input

# def forward(self, input):
#   self.last_input = input

#   h, w, c = input.shape
#   output = np.zeros((h, w , self.num_filters), dtype = self.dtype)

#   for im_region, i, j in self.iterate_regions(input):
#     for filter in range(self.num_filters):
#       output[i, j, filter] = np.sum(im_region * self.filters[filter,:,:,:], axis=(0, 1, 2))

#   self.last_output = output
#   if self.activation is not None:
#     output = self.activation(output)
#     return output



# # -*- coding: utf-8 -*-
# """conv22aa2_update.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1EvHhcnyRf0vNu82JeAekTnIhDJ9VA-9s
# """

# import numpy as np

# class Conv3x3_1_to_n_padding:
#     def __init__(self, output=1, input=1, activation=None, dtype=np.float32):
#         num_filters = output
#         in_ch = input
#         self.activation = activation
#         self.dtype = dtype
#         self.num_filters = num_filters
#         self.in_ch = in_ch 

#         # filters is a 3d array with dimensions (num_filters, 3, 3)
#         # We divide by 9 to reduce the variance of our initial values
#         self.filters = np.random.randn(num_filters, 3, 3).astype(dtype) / 9

#     def initialize_weights(self):
#         self.filters = np.random.randn(self.num_filters, 3, 3).astype(self.dtype) / 9

#     def iterate_regions(self, image):
#         '''
#         Generates all possible 3x3 image regions using valid padding.
#         - image is a 2d numpy array.
#         '''
#         image_padded = np.pad(image, 1, mode="constant", constant_values=0)
#         h, w = image_padded.shape[:2]
#         for i in range(h - 2):
#             for j in range(w - 2):
#                 im_region = image_padded[i:(i + 3), j:(j + 3)]
#                 yield im_region, i, j

                
#     def forward(self, input):
#         self.last_input = input
#         if input.ndim == 2:  # Handle 2D input
#             input = np.expand_dims(input, axis=-1)
#             h, w, _ = input.shape
#             output = np.zeros((h - 2, w - 2, self.num_filters))
#             for im_region, i, j in self.iterate_regions(input):
#                 output[i, j] = np.sum(im_region * self.filters, axis=(0, 1))
#                 self.last_output = output
#                 if self.activation is not None:
#                     output = self.activation(output)
#                     return output
            


        
# #     def forward(self, input):
# #       print('Input shape:', input.shape)  # Add this line
# #       self.last_input = input
# #       h, w = input.shape
# #       output = np.zeros((h, w, self.num_filters), dtype=self.dtype)
# #       for im_region, i, j in self.iterate_regions(input):
# #         output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
# #         self.last_output = output
# #         if self.activation is not None:
# #           output = self.activation(output)
# #           return output

#     def backprop(self, d_L_d_out, learn_rate):
#         '''
#         Performs a backward pass of the conv layer.
#         - d_L_d_out is the loss gradient for this layer's outputs.
#         - learn_rate is a float.
#         '''
#         d_L_d_filters = np.zeros(self.filters.shape, dtype=self.dtype)
#         d_L_d_input = np.zeros(self.last_input.shape, dtype=self.dtype)
#         if self.activation is not None:
#             d_L_d_out = self.activation.backward(self.last_output) * d_L_d_out
#         for im_region, i, j in self.iterate_regions(self.last_input):
#             for f in range(self.num_filters):
#                 d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
#                 for out_ch in range(self.num_filters):
#                     d_L_d_input[:, :] += d_L_d_out[:, :, out_ch] * self.filters[f]
#         # Update filters
#         self.filters -= learn_rate * d_L_d_filters
#         return d_L_d_input

#     def get_weights_fp16(self):
#       return self.filters
#     def set_weights(self, weights):
#       self.weights = weights




# class Conv3x3_n_to_n_padding:
#     def __init__(self, output=1, input=1, activation=None, dtype=np.float32):
#         num_filters = output
#         in_ch = input
#         self.activation = activation
#         self.dtype = dtype
#         self.num_filters = num_filters
#         self.in_ch = in_ch

#         # filters is a 4d array with dimensions (num_filters, 3, 3, in_ch)
#         # We divide by 9 to reduce the variance of our initial values
#         self.filters = np.random.randn(num_filters, 3, 3, in_ch).astype(dtype) / 9

#     def initialize_weights(self):
#         self.filters = np.random.randn(self.num_filters, 3, 3, self.in_ch).astype(self.dtype) / 9

#     def iterate_regions(self, image):
#         '''
#         Generates all possible 3x3 image regions using valid padding.
#         - image is a 2d numpy array.
#         '''
#         image_padded = np.pad(image, 1, mode="constant", constant_values=0)[:, :, 1:-1]
#         h, w = image_padded.shape[:2]
#         for i in range(h - 2):
#             for j in range(w - 2):
#                 im_region = image_padded[i:(i + 3), j:(j + 3), :]
#                 yield im_region, i, j

                
#     def forward(self, input):
#         self.last_input = input
#         if input.ndim == 2:  # Handle 2D input
#             input = np.expand_dims(input, axis=-1)
#             h, w, _ = input.shape
#             output = np.zeros((h - 2, w - 2, self.num_filters))
#             for im_region, i, j in self.iterate_regions(input):
#                 for f in range(self.num_filters):
#                     output[i, j, f] = np.sum(im_region * self.filters[f], axis=(0, 1, 2))
#                     self.last_output = output
#                     if self.activation is not None:
#                         output = self.activation(output)
#                         return output
            
# #     def forward(self, input):
# #         print('Input shape:', input.shape)  # Add this line

# #         '''
# #         Performs a forward pass of the conv layer using the given input.
# #         Returns a 3d numpy array with dimensions (h, w, num_filters).
# #         - input is a 3d numpy array
# #         '''
# #         self.last_input = input
# #         h, w, c = input.shape
# #         output = np.zeros((h, w, self.num_filters), dtype=self.dtype)
# #         for im_region, i, j in self.iterate_regions(input):
# #             for f in range(self.num_filters):
# #                 output[i, j, f] = np.sum(im_region * self.filters[f], axis=(0, 1, 2))
# #         self.last_output = output
# #         if self.activation is not None:
# #             output = self.activation(output)
# #         return output.astype(self.dtype)

#     def backprop(self, d_L_d_out, learn_rate):
#         d_L_d_filters = np.zeros(self.filters.shape, dtype=self.dtype)
#         for im_region, i, j in self.iterate_regions(self.last_input):
#             for f in range(self.num_filters):
#                 d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
#         d_L_d_input = np.zeros(self.last_input.shape, dtype=self.dtype)
#         # Method 4
#         for im_region, i, j in self.iterate_regions(d_L_d_out):
#             for in_ch in range(d_L_d_input.shape[-1]):
#                 d_L_d_input[i, j, in_ch] += np.sum(
#                     np.matmul(im_region[:, :, :], np.transpose(self.filters[:, :, :, in_ch], axes=(2, 0, 1))),
#                     axis=(0, 1, 2))
#         # Apply activation derivative
#         if self.activation is not None:
#             d_L_d_input = d_L_d_input * self.activation.derivative(self.last_output)
#         self.filters -= learn_rate * d_L_d_filters
#         return d_L_d_input

#     def get_weights_fp16(self):
#       return self.filters
#     def set_weights(self, weights):
#       self.weights = weights
