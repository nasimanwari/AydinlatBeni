import requests
from flask import Flask, render_template, redirect, url_for, request
from datetime import datetime, timedelta
import time
import json
import os
import waitress

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])

def update_img():
   return render_template('display.html')
  

if __name__ == "__main__":
     app.debug = False
     port = int(os.environ.get('PORT', 33507))
     waitress.serve(app, port=port)