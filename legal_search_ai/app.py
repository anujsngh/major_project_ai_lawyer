from flask import Flask, request, jsonify, render_template, redirect, url_for
from main import *
import logging
import json

# Set up logging
logging.basicConfig(filename='data/logs.log', level=logging.DEBUG)


app = Flask(__name__)


# API endpoint for handling search requests

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html", request=request)



# @app.route('/search', methods=['GET'])
# def search():
#     return redirect(url_for('home'))


@app.route('/search/rel_cases', methods=['GET', 'POST'])
def search_rel_cases():
    if request.method == "POST":
        rel_case_query = request.form.get("query")
        top_docs = rel_cases_search(rel_case_query)
        # top_docs = json.loads(top_docs_json)
        # top_documents_json = run1(rel_case_query)
        # top_docs = jsonify(top_documents_json)
        return render_template("result_rel_cases.html", top_docs=top_docs)
    return render_template("search_rel_cases.html")


@app.route('/search/rel_statutes', methods=['GET', 'POST'])
def search_rel_statutes():
    if request.method == "POST":
        rel_statutes_query = request.form.get("query")
        top_docs = rel_statutes_search(rel_statutes_query)
        # return statutes
        return render_template("result_rel_statutes.html", top_docs=top_docs)
    return render_template("search_rel_statutes.html")


@app.route('/search/statute', methods=['GET', 'POST'])
def search_statute():
    if request.method == "POST":
        statutes_query = request.form.get("query")
        top_docs = statute_search(statutes_query)
        return render_template("result_statutes.html", top_docs=top_docs)
    return render_template("search_statutes.html")


@app.route('/search/case', methods=['GET', 'POST'])
def search_case():
    judgement_urls = case_search()
    return render_template("result_cases.html", urls=judgement_urls)






if __name__ == '__main__':
    app.run(debug=True)