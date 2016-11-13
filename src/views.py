"""
Routes and views for the flask application.
"""

from datetime import datetime
import time

from flask import render_template
from src import app, api


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home',
        year=datetime.now().year
    )


@app.route('/result')
def result():
    """Renders the result page."""
    return render_template(
        'result.html',
        title='Result',
        year=datetime.now().year,
        current_time=time.time()
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
