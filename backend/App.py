from flask import Flask, request, jsonify
from utils import extract_keywords_run1, extract_keywords_run2, extract_keywords_and_vectors_run3, retrieve_documents_bm25, retrieve_documents_word2vec, combine_results, load_case_docs


app = Flask(__name__)

# API endpoint for handling search requests
@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('query')
    task = request.args.get('task')
    
    # Process search query and task, and retrieve search results
    # Replace this with your actual backend logic

    ### For Task 1 ###

    # Extract keywords from the query for Run 1
    query_keywords_run1 = extract_keywords_run1(query)

    # Extract keywords from the query for Run 2
    query_keywords_run2 = extract_keywords_run2(query)

    # Extract keywords and compute word2vec vectors for Run 3
    query_keywords_run3, query_vector_run3 = extract_keywords_and_vectors_run3(query)

    # Retrieve documents using BM25 for Run 1 and Run 2
    documents_bm25_run1 = retrieve_documents_bm25(query_keywords_run1)
    documents_bm25_run2 = retrieve_documents_bm25(query_keywords_run2)

    # Retrieve documents using word2vec and Euclidean distance for Run 3
    documents_word2vec_run3 = retrieve_documents_word2vec(query_keywords_run3, query_vector_run3)

    # Combine the results from all three runs
    merged_documents = combine_results(documents_bm25_run1, documents_bm25_run2, documents_word2vec_run3)

    case_documents = load_case_docs()


    # Retrieve the top 3 ranked case documents
    top_docs_indices = [doc_idx for doc_idx, dist in merged_documents[:3]]  # Get indices of top 3 documents
    top_documents = [case_documents[doc_idx] for doc_idx in top_docs_indices]  # Get top 3 documents

    # Create a list of dictionaries to represent the top documents
    top_documents_json = []
    for i, document in enumerate(top_documents):
        doc_dict = {"document_id": i+1, "content": document}  # Create a dictionary for each document
        top_documents_json.append(doc_dict)  # Add the dictionary to the list

    # Create a JSON object
    # json_obj = json.dumps(top_documents_json, indent=4)  # Convert the list of dictionaries to JSON

    # print(json_obj)  # Print the JSON object

    search_results = top_documents_json

    # Placeholder search results for demonstration purposes
    # search_results = [
    #     {
    #         'id': 1,
    #         'title': 'Result 1 Title',
    #         'summary': 'Result 1 Summary'
    #     },
    #     {
    #         'id': 2,
    #         'title': 'Result 2 Title',
    #         'summary': 'Result 2 Summary'
    #     },
    #     {
    #         'id': 3,
    #         'title': 'Result 3 Title',
    #         'summary': 'Result 3 Summary'
    #     }
    # ]
    
    return jsonify(search_results)

if __name__ == '__main__':
    app.run(debug=True)
