"""
Routes and views for the internal API.
"""

from flask import request, jsonify
import numpy as np

from src import app

cached_images = []
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/upload-images', methods=['POST'])
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def get_image_values(file_storage):
    content = file_storage.read()
    return content
