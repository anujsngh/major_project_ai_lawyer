from flask import Flask, request, jsonify, render_template, redirect, url_for
from main import run1
import logging
import json

# Set up logging
logging.basicConfig(filename='data/logs.log', level=logging.DEBUG)


app = Flask(__name__)


# API endpoint for handling search requests

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template("index.html")



# @app.route('/search', methods=['GET'])
# def search():
#     return redirect(url_for('home'))


@app.route('/search/rel_cases', methods=['GET', 'POST'])
def search_rel_cases():
    if request.method == "POST":
        rel_case_query = request.form.get("query")
        top_docs = run1(rel_case_query)
        # top_docs = json.loads(top_docs_json)
        # top_documents_json = run1(rel_case_query)
        # top_docs = jsonify(top_documents_json)
        return render_template("result_rel_cases.html", top_docs=top_docs)
    return render_template("search_rel_cases.html")


@app.route('/search/rel_statutes', methods=['GET', 'POST'])
def search_rel_statutes():
    return redirect(url_for('home'))


@app.route('/search/statute', methods=['GET', 'POST'])
def search_statute():
    return redirect(url_for('home'))


@app.route('/search/case', methods=['GET', 'POST'])
def search_case():
    return redirect(url_for('home'))






if __name__ == '__main__':
    app.run(debug=True)