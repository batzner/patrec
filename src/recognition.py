import random

import numpy as np
import cv2 as cv
from src import app
from math import sqrt
image_resolution = 250000
bag_of_words_size = 20

class PatternRecognition(object):
    
    def __init__(self, images, pattern):
        # convert to numpy array and equilize image size so that each image has the same number of pixels.
        self.images_color = [equalize_image_size(np.array(image), image_resolution) for image in images]
        self.pattern_color = equalize_image_size(np.array(pattern), image_resolution)
        self.images_gray = self.images_color
        self.pattern_gray = self.pattern_color
        # convert to grayscale for SIFT
        #self.images_gray = [cv.cvtColor(image, cv.COLOR_RGB2GRAY) for image in self.images_color]
        #self.pattern_gray = cv.cvtColor(pattern_color, cv.COLOR_RGB2GRAY)

        self.sift = cv.SIFT()
        self.bow = None
        self.pattern_sift_keypoints = []
        self.pattern_sift_descriptors = []
        self.pattern_bow_features = []


    def initialize(self):
        # calculate a bag of words for the given images.
        self.bow = self.__calculate_bow_for_images()    

        # compute SIFT keypoints and bag of words feature for the pattern.
        # this is used to calculate a similarity score between the query_images and the pattern.
        # however, the effectiveness depends on the number of images that are used.        
        self.pattern_sift_keypoints, self.pattern_sift_descriptors = self.sift.detectAndCompute(self.pattern_gray, None)
        self.pattern_bow_features = self.bow.compute_feature_vector(self.pattern_gray, self.pattern_sift_keypoints)
        return True

    def is_match(self, image):
        """ 
        Tries to find the pattern for the given query_image.
        Returns True, [corners] if pattern is matched and False, None if the query_image does not contain the pattern.
        """

        #calculate SIFT keypoints and descriptors for query_image
        img_sift_kp, img_sift_desc = self.sift.detectAndCompute(image, None)

        sift_similarity_score = self.__calculate_sift_similarity(image, img_sift_kp)

        # find matches with BruteForce Matcher and get the two best matches.
        # distance measurement is Norm_L2 which is the euclidean distance.
        bf = cv.BFMatcher()
        matches = bf.knnMatch(img_sift_desc, self.pattern_sift_descriptors, k=2)        

        # apply ratio test as reccomended in D. G. Lowe. "Object recognition from local scale-invariant features."
        # ratio is slightly altered from original paper.
        good_matches = []
        matched_keypoints = []
        distances = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                distances.append(m.distance)
                good_matches.append([m])
                matched_keypoints.append(img_sift_kp[m.queryIdx])

        match_ratio_score = float(len(good_matches)) / (float(len(matches)) + 1e-10) # 1e-10 to prevent 0-div
        mean_distance_score = np.mean(distances)

        if (match_ratio_score > 0.023 and mean_distance_score < 200 and sift_similarity_score < 0.6) or (match_ratio_score > 0.05) or (mean_distance_score < 140) or (sift_similarity_score < 0.35):
            # match            
            return (True, {'match_ratio_score': match_ratio_score, 'mean_distance_score': mean_distance_score, 'sift_similarity_score': sift_similarity_score})
        else:
            # no match
            return (False, {'match_ratio_score': match_ratio_score, 'mean_distance_score': mean_distance_score, 'sift_similarity_score': sift_similarity_score})

    def next(self):
        for image_gray in self.images_gray:
            yield self.is_match(image_gray)



    def __calculate_sift_similarity(self, image, img_sift_kp):
        # get bow features for image
        img_bow_sift_desc = self.bow.compute_feature_vector(image, img_sift_kp)
        return self.__calculate_chi2_distance(self.pattern_bow_features, img_bow_sift_desc)

    def __calculate_chi2_distance(self, f1, f2, eps = 1e-10):
        d = np.sum([ ((a-b)**2) / (a + b + eps) for (a, b) in zip(f1, f2)])
        return d

    def __calculate_bow_for_images(self):

        app.logger.debug('Setting up descriptor dictionary.')
        
        # calculate SIFT descriptors for each image
        descriptors = []
        for image in self.images_gray:
            descriptors.append(self.sift.detectAndCompute(image, None)[1])
        descriptors.append(self.sift.detectAndCompute(self.pattern_gray, None)[1])

        bow = BagOfWords(bag_of_words_size)
        bow.create_BOW(descriptors)

        return bow

def find_matches(images, pattern):
    result = []
    pr = PatternRecognition(images, pattern)
    pr.initialize()    
    
    # Search for pattern matches
    for is_match, scores in pr.next():

        if is_match:
            app.logger.info('[X] Match.\tMatch Ratio: {0:.3f} - Mean Distance: {1:.3f} - Sift Bow Similarity: {2:.3f}'.format(scores['match_ratio_score'], scores['mean_distance_score'], scores['sift_similarity_score']))
        else:
            app.logger.info('[-] No Match.\tMatch Ratio: {0:.3f} - Mean Distance: {1:.3f} - Sift Bow Similarity: {2:.3f}'.format(scores['match_ratio_score'], scores['mean_distance_score'], scores['sift_similarity_score']))
        
        result.append({'value': scores['match_ratio_score'], 'is_match': is_match})

    return result


def equalize_image_size(image, size):
    """ Resizes the image to fit the given size."""
    #print image.shape
    # image size
    w, h = (image.shape[1], image.shape[0])
        
    if (w*h) != size:
        # calculate factor so that we can resize the image. The total size of the image (w*h) should be ~ size.
        # w * x * h * x = size
        ls = float(h * w)
        ls = float(size) / ls
        factor = sqrt(ls)
        image = cv.resize(image, (0,0), fx=factor, fy=factor)
   
    return image



class BagOfWords(object):
    """Wrapper for openCV Bag of words logic."""
        
    def __init__(self, size, dextractor='SIFT', dmatcher='FlannBased', vocabularyType=np.float32):
        self.size = size
        self.dextractor = dextractor
        self.dmatcher = dmatcher
        self.vocabularyType = vocabularyType

    def create_BOW(self, descriptors):
        """Computes a Bag of Words with a set of descriptors."""

        app.logger.debug('Creating BOW with size {0} with {1} descriptors.'.format(self.size, len(descriptors)))
        bowTrainer = cv.BOWKMeansTrainer(self.size)

        # Convert the list of numpy arrays to a single numpy array
        npdescriptors = np.concatenate(descriptors)

        # an OpenCV BoW only takes floats as descriptors. convert if necessary
        if not npdescriptors.dtype == np.float32:
            npdescriptors = np.float32(npdescriptors)

        app.logger.debug('Clustering BOW with Extractor {0} and Matcher {1}'.format(self.dextractor, self.dmatcher))
        self.__BOWVocabulary = bowTrainer.cluster(npdescriptors)

        # need to convert vocabulary?
        if self.__BOWVocabulary.dtype != self.vocabularyType:
            self.__BOWVocabulary = self.__BOWVocabulary.astype(self.vocabularyType)

        app.logger.debug('BOW vocabulary creation finished.')

        # Create the BoW descriptor
        self.__BOWDescriptor = cv.BOWImgDescriptorExtractor(cv.DescriptorExtractor_create(self.dextractor), cv.DescriptorMatcher_create(self.dmatcher))
        self.__BOWDescriptor.setVocabulary(self.__BOWVocabulary)

    def compute_feature_vector(self, image, keypoints):
        return self.__BOWDescriptor.compute(image, keypoints)  