"""
Routes and views for the internal API.
"""
from PIL import Image
from flask import request, jsonify
import numpy as np

from src import app, recognition

ROUTE_PREFIX = '/api'

cached_images = []
cached_pattern = None
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def empty_cache():
    global cached_images, cached_pattern
    cached_images = []
    cached_pattern = None


@app.route(ROUTE_PREFIX+'/upload-images', methods=['POST'])
def upload_images():
    app.logger.debug('Upload image called')

    files = request.files.getlist('file[]')
    if not files:
        files = [request.files['file']]
    for file_storage in files:
        app.logger.debug('Filename: %s' % file_storage.filename)

        if not file_storage or not allowed_file(file_storage.filename):
            app.logger.info('Extension name error')
            return jsonify(status=400, error='Wrong extension. Only images (%s) are supported.' %
                                             app.config['ALLOWED_EXTENSIONS'])

        # Convert the FileStorage to a numpy array and add it to the cache
        image = get_image_values(file_storage)
        cached_images.append(image)

    return jsonify(status=200, success='true')


@app.route(ROUTE_PREFIX+'/upload-pattern', methods=['POST'])
def upload_pattern():
    global cached_pattern
    app.logger.debug('Upload pattern called')
    file_storage = request.files['file']
    app.logger.debug('Filename: %s' % file_storage.filename)

    if not file_storage or not allowed_file(file_storage.filename):
        app.logger.info('Extension name error')
        return jsonify(status=400, error='Wrong extension. Only images (%s) are supported.' %
                                         app.config['ALLOWED_EXTENSIONS'])

    # Convert the FileStorage to a numpy array and add it to the cache
    cached_pattern = get_image_values(file_storage)
    return jsonify(status=200, success='true')


@app.route(ROUTE_PREFIX+'/find-matches')
def find_matches():
    result = recognition.find_matches(images=cached_images, pattern=cached_pattern)
    return jsonify(result)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def get_image_values(file_storage):
    # Open the stream and convert it to grayscale
    image = Image.open(file_storage).convert('L')
    values = np.asarray(image)
    return values
