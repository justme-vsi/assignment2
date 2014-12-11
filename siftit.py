import Image
import os
import numpy as np
import scipy.cluster.vq as vq
from os import walk
from scipy.spatial import distance as dist

#config
root_dir = "image_db"
tmp_sift_file = 'tmp_sift.txt'
codebook_file = 'codebook.txt'
matrix_file = 'matrix.txt'
matrix_whitend_file = 'matrix_whitend.txt'
centroids_file ='centroids.txt'
obs_file = 'obs.txt'

def process_image(imagename, params="--edge-thresh 10 --peak-thresh 5"):
    """ Process an image and save the results in a file. """
    if imagename[-3:] != 'pgm':
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
    cmmd = str("sift " + imagename + " --output=" + tmp_sift_file + " " + params)
    os.system(cmmd)


def read_features_from_file(filename):
    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]  # feature locations, descriptors


def get_concat_results():
    for (dirpath, dirnames, filenames) in walk(root_dir):
        if dirpath == root_dir:
            continue
        result = np.ones((1, 128))
        training_image_count = int(len(filenames) * 0.1)
        if training_image_count == 0:
            training_image_count = 1
        for file in filenames[0:training_image_count]:
            if file[-3:].upper() != "JPG":
                continue
            # image_path = dirpath + "/" + file
            # process_image(image_path)
            result = np.vstack((result, read_features_from_file(tmp_sift_file)[1]))
        file_path = dirpath + '/' + dirpath.split('/')[1] + '.txt'
        np.savetxt(file_path, result[1:])  #remove initialize vector

# get_concat_results()
# matrix = np.loadtxt(matrix_file)
# # np.savetxt(obs_file, vq.whiten(matrix))
# matrix_whitend = np.loadtxt(obs_file)
# # code_book = vq.kmeans(obs, 666)[0]
# # np.savetxt(codebook_filename, code_book)
# codebook = np.loadtxt(codebook_file)
# # np.savetxt(centroids_file, vq.vq(obs, codebook))
centroids = np.loadtxt(centroids_file)
# print matrix_file, matrix.shape
# print matrix_whitend_file, matrix_whitend.shape
# print codebook_file, codebook.shape
# print centroids_file, centroids.shape
# print '\n'
#
# print codebook[centroids[0][0]]
print centroids[0][0]
print centroids[1][0]


def get_histogram(vq_res):

    hist = np.zeros((1,len(vq_res[0])))
    print vq_res[0]
    print hist.shape
    for code in vq_res[0]:
        print code
        #print hist[code]
        hist[0][code]+=1
        print hist[0][code]

    print hist.reshape(-1)

    return hist.reshape(-1)

def compare_hist(hist1, hist2):

    return dist.euclidean(hist1,hist2)


def create_hists(descs, code_book, hist_filename):

    obs = vq.whiten(descs)
    centroids = vq.vq(obs, code_book)
    hist = get_histogram(centroids)
    np.savetxt(hist_filename+".txt", hist)
    return hist