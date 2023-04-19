from flask import Flask, request, jsonify
from main import run1
import logging

# Set up logging
logging.basicConfig(filename='data/logs.log', level=logging.DEBUG)


app = Flask(__name__)


# API endpoint for handling search requests
@app.route('/api/search', methods=['GET'])
def search():
    # return "hello"
    query = request.args.get('query')
    task = request.args.get('task')

    search_results = run1(query)

    return jsonify(search_results)


if __name__ == '__main__':
    app.run(debug=True)