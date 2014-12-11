import siftit
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
work_dir = 'image_db/accordion'
# siftit.get_descriptor_matrix_for_dir(work_dir)
# training_matrix = np.loadtxt('image_db/accordion/accordion.txt')
# whitend_matrix_file, whitend_matrix = siftit.generate_witend_matrix(training_matrix)
# codebook_file, codebook = siftit.genrate_kmean(whitend_matrix, 50)

# codebook = np.loadtxt(siftit.codebook_file)
# print codebook.shape
# siftit.do_histograms(work_dir, codebook)

# siftit.generate_vq(whitend_matrix, codebook)
hist = np.loadtxt('image_db/accordion/image_0031.jpg.txt')
print hist.shape
plt.hist(hist, 128)
plt.show()
