import imp
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy
import pandas as pd

app = Flask(__name__) # Initialize the flask App
model = pickle.load(open('')) # load model here