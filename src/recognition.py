import random

import numpy as np
import cv2 as cv
from src import app
from math import sqrt
from sklearn import metrics

image_resolution = 250000
bag_of_words_size = 20
histogram_image_segments = 16
histogram_number_of_bins = 64
histogram_color_range = [0, 256]
histogram_scale_hist = True
sliding_window = True


class PatternRecognition(object):
	def __init__(self, images, pattern):
		# convert to numpy array and equilize image size so that each image has the same number of pixels.
		self.images_color = [equalize_image_size(np.array(image), image_resolution) for image in images if
		                     not image is None and len(image) > 0]
		self.pattern_color = equalize_image_size(np.array(pattern), image_resolution)
		# self.images_gray = self.images_color
		# self.pattern_gray = self.pattern_color
		# convert to grayscale for SIFT
		self.images_gray = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in self.images_color]
		self.pattern_gray = cv.cvtColor(self.pattern_color, cv.COLOR_BGR2GRAY)

		self.sift = cv.SIFT()
		self.bow = None
		self.pattern_sift_keypoints = []
		self.pattern_sift_descriptors = []
		self.pattern_bow_features = []
		self.pattern_color_features = []

	def initialize(self):
		# calculate a bag of words for the given images.
		self.bow = self.__calculate_bow_for_images()

		# compute SIFT keypoints and bag of words feature for the pattern.
		# this is used to calculate a similarity score between the query_images and the pattern.
		# however, the effectiveness depends on the number of images that are used.
		self.pattern_sift_keypoints, self.pattern_sift_descriptors = self.sift.detectAndCompute(self.pattern_gray, None)
		self.pattern_bow_features = self.bow.compute_feature_vector(self.pattern_gray, self.pattern_sift_keypoints)
		self.pattern_color_features = self.__create_color_histogram_feature_vector(self.pattern_color)
		return True

	def is_match(self, image_color, mark_match=True):
		"""
        Tries to find the pattern for the given query_image.
        Returns True, [corners] if pattern is matched and False, None if the query_image does not contain the pattern.
        """

		image_gray = cv.cvtColor(image_color, cv.COLOR_RGB2GRAY)

		# calculate SIFT keypoints and descriptors for query_image
		img_sift_kp, img_sift_desc = self.sift.detectAndCompute(image_gray, None)

		# if no sift keypoints could be found there is no match
		if len(img_sift_kp) == 0:
			return (False, {'match_ratio_score': 0, 'mean_distance_score': 10000, 'sift_similarity_score': 10000,
			                'color_similarity_score': 10000, 'votes': 0})

		sift_similarity_score = self.__calculate_sift_score(image_gray, img_sift_kp)
		color_similarity_score = self.__calculate_color_score(image_color)

		# find matches with BruteForce Matcher and get the two best matches.
		# distance measurement is Norm_L2 which is the euclidean distance.
		bf = cv.BFMatcher()
		matches = bf.knnMatch(img_sift_desc, self.pattern_sift_descriptors, k=2)

		# apply ratio test as reccomended in D. G. Lowe. "Object recognition from local scale-invariant features."
		# ratio is slightly altered from original paper.
		good_matches = []
		matched_keypoints = []
		distances = []
		for m, n in matches:
			if m.distance < 0.75 * n.distance:
				distances.append(m.distance)
				good_matches.append([m])
				matched_keypoints.append(img_sift_kp[m.queryIdx])

		match_ratio_score = float(len(good_matches)) / (float(len(matches)) + 1e-10)  # 1e-10 to prevent 0-div

		if len(distances) > 0:
			mean_distance_score = np.mean(distances)
		else:
			mean_distance_score = 10000
		filtered_keypoints = self.__eliminate_closest_neighbor_noise(matched_keypoints)
		# vote casting
		votes = 0
		if match_ratio_score > 0.020 and len(filtered_keypoints) > 1:
			votes += match_ratio_score * 300
		if len(filtered_keypoints) > 4:
			votes += 0.3 * len(filtered_keypoints)
		if mean_distance_score < 180:
			votes += 2
		if mean_distance_score > 210:
			votes -= 4.5
		if sift_similarity_score < 0.7:
			votes += 1.5
		if color_similarity_score < 40:
			votes += 0.5

		if votes > 9:
			# match

			if mark_match:
				# filter "outlier" keypoints that are likely not part of the real object

				image_color = cv.cvtColor(image_color, cv.COLOR_RGB2BGR)
				image_color = cv.drawKeypoints(image_color, matched_keypoints)
				# cv.imshow('[{0}] - MR: {1:.3f} - Mean Dist: {2:.1f} - SI Bow Score: {3:.3f} - Color Score: {4:.3f} - Matches: {5}'.format(votes, match_ratio_score, mean_distance_score, sift_similarity_score, color_similarity_score, len(good_matches)), image_color)

			return (True, {'match_ratio_score': match_ratio_score, 'mean_distance_score': mean_distance_score,
			               'sift_similarity_score': sift_similarity_score,
			               'color_similarity_score': color_similarity_score, 'votes': votes})
		else:
			# no match
			image_color = cv.cvtColor(image_color, cv.COLOR_RGB2BGR)
			# if mark_match:
			# cv.imshow('[-] - MR: {0:.3f} - Mean Dist: {1:.1f} - SI Bow Score: {2:.3f} - Color Score: {3:.3f}'.format(match_ratio_score, mean_distance_score, sift_similarity_score, color_similarity_score), image_color)

			return (False, {'match_ratio_score': match_ratio_score, 'mean_distance_score': mean_distance_score,
			                'sift_similarity_score': sift_similarity_score,
			                'color_similarity_score': color_similarity_score, 'votes': votes})

	def next(self):
		for image in self.images_color:
			if not sliding_window:
				yield self.is_match(image)
			else:
				is_match, score = self.is_match(image)
				if is_match:
					yield is_match, score
					continue
				matches = []
				scores = []
				votes = []
				window_visualization = cv.cvtColor(image, cv.COLOR_RGB2BGR)

				for image_region, image_part in sliding_window(image, size=(100, 100), stride=100):
					# increase size of sliding window part
					image_part = equalize_image_size(image_part, image_resolution)
					is_match, score = self.is_match(image_part, True)
					scores.append(score)
					votes.append(score['votes'])
					if is_match:
						matches.append(1)
						window_visualization = image_region.overlay_rectangle(window_visualization)
					else:
						matches.append(0)
				# cv.imshow("Matches: {0} - Votes: {1}".format(np.sum(matches), np.sum(votes)), window_visualization)
				yield np.sum(matches) > 0, scores[0]

	def __calculate_sift_score(self, image, img_sift_kp):
		# if no keypoints were found return a bad sift score
		if len(img_sift_kp) == 0:
			return 10000

		# get bow features for image
		img_bow_sift_desc = self.bow.compute_feature_vector(image, img_sift_kp)
		return self.__calculate_chi2_distance(self.pattern_bow_features, img_bow_sift_desc)

	def __calculate_color_score(self, image):
		image_color_feature = self.__create_color_histogram_feature_vector(image)

		color_score = self.__calculate_chi2_distance(self.pattern_color_features, image_color_feature)
		return color_score

	def __create_color_histogram_feature_vector(self, image):
		image_cols = image_rows = sqrt(histogram_image_segments)

		# split image into x different parts. (default 4 parts)
		image_parts = []
		colors = ("r", "g", "b")
		height, width = image.shape[:2]
		part_height = int(height / image_rows)
		part_width = int(width / image_cols)
		scale_max_possible_value = part_width * part_height

		# modify height and width in case part_height and part_width don't add up to the real width and height.
		# in this case the image would be cut into more than x parts because some leftover pixels would be included.
		height = int(part_height * image_rows)
		width = int(part_width * image_cols)

		for y in xrange(0, height, part_height):
			for x in xrange(0, width, part_width):
				image_parts.append(crop_image(image, x, y, part_width, part_height))

		histogram = []
		for img in image_parts:
			for i, color in enumerate(colors):
				hist = cv.calcHist([img], [i], None, [histogram_number_of_bins], histogram_color_range)

				if histogram_scale_hist:
					# max possible value is w * h of imagePart
					hist /= scale_max_possible_value
				histogram.extend(hist)
		return np.array(np.concatenate(histogram))

	def __calculate_chi2_distance(self, f1, f2, eps=1e-10):
		try:
			d = np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(f1, f2)])
		except:
			# app.logger.debug('Could not calculate Chi2 distance: F1 {0} - F2 {1}'.format(f1, f2))
			d = 1000
		return d

	def __calculate_bow_for_images(self):

		# app.logger.debug('Setting up descriptor dictionary.')

		# calculate SIFT descriptors for each image
		descriptors = []
		for image in self.images_gray:
			descriptors.append(self.sift.detectAndCompute(image, None)[1])
		descriptors.append(self.sift.detectAndCompute(self.pattern_gray, None)[1])

		bow = BagOfWords(bag_of_words_size)
		bow.create_BOW(descriptors)

		return bow

	def __eliminate_closest_neighbor_noise(self, keypoints):
		"""
        This method tries to eliminate "outlier" keypoints by comparing the distance to their closest neighbors.
        If the distance is greater than the median - std distance the keypoints are filtered out.
        Distance is measured by Norm_L2
        """

		if len(keypoints) == 0:
			return keypoints

		# convert keypoint list to point list
		keypoint_point_list = [kp.pt for kp in keypoints]

		# compute pairwise distance
		distance_matrix = metrics.pairwise_distances(keypoint_point_list)
		d_mean = distance_matrix.mean()
		d_std = distance_matrix.std()

		# iterate over all keypoints and filter outliers
		filtered_keypoints = []
		for i in xrange(len(keypoints)):
			# get the mean distance of the 8 closest neighbors of kp or all neigbors
			k = min(len(distance_matrix[i]), 8)
			neighbor_distances = np.partition(distance_matrix[i], k - 1)[:k - 1]
			# filter 0-distance to point itself
			neighbor_distances = neighbor_distances[np.nonzero(neighbor_distances)]
			if len(neighbor_distances) == 0:
				return []
			mean_neighbor_distance = neighbor_distances.mean()

			if mean_neighbor_distance <= d_mean - d_std:
				filtered_keypoints.append(keypoints[i])

		return filtered_keypoints


# def find_matches(images, pattern):
#    result = []
#    pr = PatternRecognition(images, pattern)
#    pr.initialize()    

#    # Search for pattern matches
#    for is_match, scores in pr.next():

#        if is_match:
#            app.logger.info('[X] Match.\tMatch Ratio: {0:.3f} - Mean Distance: {1:.3f} - Sift Bow Similarity: {2:.3f}'.format(scores['match_ratio_score'], scores['mean_distance_score'], scores['sift_similarity_score']))
#        else:
#            app.logger.info('[-] No Match.\tMatch Ratio: {0:.3f} - Mean Distance: {1:.3f} - Sift Bow Similarity: {2:.3f}'.format(scores['match_ratio_score'], scores['mean_distance_score'], scores['sift_similarity_score']))

#        result.append({'value': scores['match_ratio_score'], 'is_match': is_match})
#    #cv.waitKey(0)
#    return result


def find_matches(images, pattern):
	result = []
	pr = PatternRecognition(images, pattern)
	pr.initialize()

	# Search for pattern matches
	for is_pattern, scores in pr.next():
		# cv.waitKey(0)
		# cv.destroyAllWindows()
		if is_pattern:
			result.append({'value': scores['match_ratio_score'], 'is_match': True})
			print '[X] Match.\tMatch Ratio: {0:.3f} - Mean Distance: {1:.3f} - Sift Bow Similarity: {2:.3f}'.format(
				scores['match_ratio_score'], scores['mean_distance_score'], scores['sift_similarity_score'])
		else:
			print '[-] No Match.\tMatch Ratio: {0:.3f} - Mean Distance: {1:.3f} - Sift Bow Similarity: {2:.3f}'.format(
				scores['match_ratio_score'], scores['mean_distance_score'], scores['sift_similarity_score'])
			result.append({'value': scores['match_ratio_score'], 'is_match': False})

	return result


def equalize_image_size(image, size):
	""" Resizes the image to fit the given size."""
	# print image.shape
	# image size
	w, h = (image.shape[1], image.shape[0])

	if (w * h) != size:
		# calculate factor so that we can resize the image. The total size of the image (w*h) should be ~ size.
		# w * x * h * x = size
		ls = float(h * w)
		ls = float(size) / ls
		factor = sqrt(ls)
		image = cv.resize(image, (0, 0), fx=factor, fy=factor)

	return image


def crop_image(image, x, y, w, h):
	""" Crops an image.

    Keyword arguments:
    image -- image to crop
    x -- upper left x-coordinate
    y -- upper left y-coordinate
    w -- width of the cropping window
    h -- height of the cropping window
    """

	# crop image using np slicing (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)
	image = image[y: y + h, x: x + w]
	return image


def sliding_window(image, size, stride):
	for y in xrange(0, image.shape[0], stride):
		for x in xrange(0, image.shape[1], stride):
			roi = ImageRegion(upper_left=(x, y), lower_right=(x + size[0], y + size[1]))
			yield roi, crop_image(image, x, y, size[0], size[1])


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
		self.__BOWDescriptor = cv.BOWImgDescriptorExtractor(cv.DescriptorExtractor_create(self.dextractor),
		                                                    cv.DescriptorMatcher_create(self.dmatcher))
		self.__BOWDescriptor.setVocabulary(self.__BOWVocabulary)

	def compute_feature_vector(self, image, keypoints):
		return self.__BOWDescriptor.compute(image, keypoints)


class ImageRegion(object):
	"""Class to represent an image region."""

	def __init__(self, contour=None, upper_left=None, lower_right=None, cutmask=None):
		""" Creates an image Region object.
        Region can be constructed by contour, combination of upperLeft and lowerRight or cutmask rectangle.

        Keyword arguments:
        contour -- OpenCV contour
        upperLeft -- Tuple of upperLeft Point coordinates
        lowerRight -- Tuple of lowerRight Point coordinates
        cutmask -- rectangle of region
        """

		if contour is None and cutmask is None and (upper_left is None or lower_right is None):
			raise AttributeError('Either provide upperLeft and lowerRight or a cut mask')

		self.contour = contour

		if cutmask is None and upper_left is not None and lower_right is not None:
			self.upper_left = upper_left
			self.lower_right = lower_right
			width = lower_right[0] - upper_left[0]
			height = lower_right[1] - upper_left[1]
			self.cutmask = Rectangle(upper_left, (width, height))
		elif cutmask is not None:
			self.cutmask = cutmask
			self.upper_left = cutmask.upperLeft
			self.lower_right = cutmask.lowerRight
		else:

			x, y, w, h = cv.boundingRect(np.array([contour]))

			self.cutmask = Rectangle((x, y), (w, h))
			self.upper_left = self.cutmask.upper_left
			self.lower_right = self.cutmask.lower_right

	def create_mask(self, shape):
		""" Creates a mask for the image with the image region."""

		# make sure shape is only 1 channel
		shape = (shape[0], shape[1])

		# create a new black image
		mask = np.zeros(shape, dtype=np.uint8)

		# draw the contour as a white filled area
		cv.drawContours(mask, [self.contour], 0, (255, 255, 255), thickness=-1)
		return mask

	def crop_image_region(self, image):
		""" Crops an image to the given imageRegion and returns it.
        """
		x, y = self.upper_left
		w, h = self.get_dimension()
		return crop_image(image, x, y, w, h)

	def get_roi_image(self, image):
		""" Returns the image with only the image region visible."""

		m = self.create_mask(image.shape)
		# Apply the mask
		return cv.bitwise_and(image, image, mask=m)

	def overlay_rectangle(self, image, alpha=0.1, color=(0, 255, 0)):
		"""
        Overlays a transparent rectangle.

        Keyword arguments:
        image -- image to overlay the rectangle on
        alpha -- alpha value of the rectangle
        color -- BGR color of the rectangle
        """

		result = image.copy()
		overlay = image.copy()

		cv.rectangle(overlay, self.upper_left, self.lower_right, color, -1)

		cv.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
		return result

	def get_ratio(self):
		""" Get the aspect ratio of height and width."""
		return self.cutmask.height / self.cutmask.width

	def get_dimension(self):
		""" Get the dimension of the image region."""
		return (self.cutmask.width, self.cutmask.height)


class Rectangle(object):
	""" Rectangle wrapper.
    """

	def __init__(self, upper_left_point, size):
		self.width, self.height = size
		self.upper_left = upper_left_point
		self.lower_right = (upper_left_point[0] + self.width, upper_left_point[1] + self.height)
