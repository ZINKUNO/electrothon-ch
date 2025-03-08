from flask import Flask, send_file
import os
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

@app.route('/')
def landing_page():
    return send_file('index.html')

if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=3000) 