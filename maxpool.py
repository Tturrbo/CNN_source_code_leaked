import numpy as np

np.random.seed(42)

class Maxpool2x2:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        h, w = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))
        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(img_region, axis=(0,1))
        return output
    
    def backprop(self, dL_dout):
        dL_dinput = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            dL_dinput[i*2 + i2, j*2 + j2, f2] = dL_dout[i, j ,f2]
        return dL_dinput