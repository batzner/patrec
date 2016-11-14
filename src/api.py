"""
Routes and views for the internal API.
"""
import uuid
from StringIO import StringIO

from PIL import Image
from flask import request, jsonify, send_file
import numpy as np

from src import app, recognition

ROUTE_PREFIX = '/api'

cached_images = dict()
ordered_image_ids = []
cached_pattern = None
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


@app.route(ROUTE_PREFIX + '/empty-cache', methods=['POST'])
def empty_cache():
    global cached_images, cached_pattern, ordered_image_ids
    cached_images = dict()
    ordered_image_ids = []
    cached_pattern = None
    return jsonify(status=200, success=True)


@app.route(ROUTE_PREFIX + '/upload-images', methods=['POST'])
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

        # Convert the FileStorage to an image and add it to the cache with a generated id
        image = get_pil(file_storage)
        image_id = uuid.uuid4()
        cached_images[image_id] = image
        ordered_image_ids.append(image_id)

    return jsonify(status=200, success=True)


@app.route(ROUTE_PREFIX + '/upload-pattern', methods=['POST'])
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
    cached_pattern = get_pil(file_storage)
    return jsonify(status=200, success=True)


@app.route(ROUTE_PREFIX + '/find-matches')
def find_matches():
    image_values = [get_image_values(cached_images[image_id]) for image_id in ordered_image_ids]
    pattern_values = get_image_values(cached_pattern)
    result = recognition.find_matches(images=image_values, pattern=pattern_values)

    # For each image, add its id to the result
    for image_result, image_id in zip(result, ordered_image_ids):
        image_result['id'] = image_id

    # Order the result by score
    result = sorted(result, key=lambda r: -r['value'])
    return jsonify(result)


@app.route(ROUTE_PREFIX + '/get-pattern')
def get_pattern():
    return serve_pil_image(cached_pattern)


@app.route(ROUTE_PREFIX + '/get-image/<uuid:image_id>')
def get_image(image_id):
    return serve_pil_image(cached_images[image_id])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def get_pil(file_storage):
    return Image.open(file_storage).convert('RGB')


def get_image_values(image):
    values = np.asarray(image)
    return values


def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
