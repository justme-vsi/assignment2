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
whitend_matrix_file = 'matrix_whitend.txt'
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


def read_features_from_file_last_sift():
    f = np.loadtxt(tmp_sift_file)
    return f[:, :4], f[:, 4:]  # feature locations, descriptors

def get_descriptor_matrix_for_dir(dir):
    for (dirpath, dirnames, filenames) in walk(dir):
        if dirpath == root_dir:
            continue
        result = np.ones((1, 128))
        training_image_count = int(len(filenames) * 0.1)
        if training_image_count == 0:
            training_image_count = 1
        for file in filenames[0:training_image_count]:
            if file[-3:].upper() != "JPG":
                continue
            image_path = dirpath + "/" + file
            process_image(image_path)
            result = np.vstack((result, read_features_from_file_last_sift()[1]))
        file_path = dirpath + '/' + dirpath.split('/')[1] + '.txt'
        np.savetxt(file_path, result[1:])  #remove initialize vector

def genrate_kmean(obs, k):
    codebook = vq.kmeans(obs, k)[0]
    np.savetxt(codebook_file, codebook)
    return codebook_file, codebook

def generate_vq(obs, codebook):
    centroids = vq.vq(obs, codebook)[0]
    np.savetxt(centroids_file, centroids)
    return centroids_file, centroids

def generate_witend_matrix(training_matrix):
    witend_matrix = vq.whiten(training_matrix)
    np.savetxt(whitend_matrix_file, witend_matrix)
    return whitend_matrix_file, witend_matrix

def do_histograms(dir, codebook):
    for (dirpath, dirnames, filenames) in walk(dir):
        for file in filenames:
            if file[-3:].upper() != "JPG":
                continue
            image_path = dirpath + "/" + file
            create_hists(image_path, codebook)

def get_histogram(vq_res):
    hist = np.zeros((1,len(vq_res[0])))
    print hist.shape
    for code in vq_res[0]:
        print code
        hist[0][code]+=1
        print hist[0][code]

    return hist.reshape(-1)

def compare_hist(hist1, hist2):
    return dist.euclidean(hist1,hist2)

def create_hists(image_path, code_book):
    process_image(image_path)
    descs = read_features_from_file_last_sift()[1]
    obs = vq.whiten(descs)
    centroids = vq.vq(obs, code_book)
    hist = get_histogram(centroids)
    file_name = image_path+".txt"
    np.savetxt(file_name, hist)
    return file_name, hist


