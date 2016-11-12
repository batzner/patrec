"""
Routes and views for the flask application.
"""

from datetime import datetime

import cv2 as cv

from flask import render_template
from src import app, api

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    # Empty the internal cache
    api.empty_cache()
    return render_template(
        'index.html',
        title=cv.__version__,
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
