import numpy as np

np.random.seed(42)

class Conv3x3:
    def __init__(self, n_filters):
        self.num_filters = n_filters
        self.filters = np.random.randn(n_filters, 3, 3) / 9
    
    def patches(self, image):
        h, w = image.shape
        for i in range(h-2):
            for j in range(w-2):
                img_region = image[i:(i+3), j:(j+3)]
                yield img_region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for img_region, i, j in self.patches(input):
            output[i, j] = np.sum(img_region * self.filters, axis=(1,2))
        return output
    
    def backprop(self, dL_dout, learn_rate):
        dL_dfilters = np.zeros(self.filters.shape)
        for im_region, i, j in self.patches(self.last_input):
            for f in range(self.num_filters):
                dL_dfilters[f] += dL_dout[i, j, f] * im_region
        self.filters -= learn_rate * dL_dfilters
        return None
    

