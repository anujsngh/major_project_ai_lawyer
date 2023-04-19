from utils import *
import json


def run1(query):
    documents = load_raw_case_docs()

    try:
        preprocessed_docs = load_preprocessed_docs()
    except:
        preprocessed_docs = store_preprocessed_docs(documents=documents)
    
    preprocessed_query = preprocess_text(text=query)

    # tokenized_docs = load_tokenized_docs(preprocessed_docs)

    # corpus = load_corpus(tokenized_docs=tokenized_docs)

    try:
        query_tfidf_matrix, tfidf_matrix = load_TF_IDF_feature_vectors()
    except:
        query_tfidf_matrix, tfidf_matrix = store_TF_IDF_feature_vectors(preprocessed_query, preprocessed_docs)

    similarity_scores = compute_cosine_similarity_score(query_tfidf_matrix, tfidf_matrix)

    top_docs = load_top_documents(similarity_scores, documents)

    # Create a list of dictionaries to represent the top documents
    top_documents_json = []
    for i, document in enumerate(top_docs):
        doc_dict = {"document_id": i+1, "content": document}  # Create a dictionary for each document
        top_documents_json.append(doc_dict)  # Add the dictionary to the list

    return top_documents_json
    
    


if __name__ == "__main__":
    query = "The appellant on February 9, 1961 was appointed as an Officer in Grade III in the respondent Bank ( for short 'the Bank'). He was promoted on April 1, 1968 to the Grade officer in the Foreign Exchange Department in the Head Office of the Bank. Sometime in 1964, MCH Society ( for short 'the Society') was formed of which the appellant was one of the chief promoters and thereafter its Secretary. The object of the Society was to construct residential premises for the employees of the Bank and its other members. It appears that the complaint was received in respect of the affairs of the Society relating to misappropriation of the funds of the Society and consequently, in exercise of the powers under Section S of Act A1, the Registrar on April 23, 1969 instituted an inquiry thereof. P1 was appointed the Registrar's nominee who on October 4, 1969; submitted the report holding the appellant and two other office bearers of the Society negligent in dealing with the funds of the Society causing a loss to the tune of Rs. 3,59,000/-. The Registrar on October 21, 1969, passed an order appointing an officer under Section S of A1 to assess the loss caused to the Society. However, the Government by its order dated November 29, 1969 annulled the Registrar's order dated April 23, 1969 and October 21, 1969 and directed a fresh inquiry into the affairs of the Society. On December 17, 1969, the Bank issued show cause notice to the appellant to explain within fifteen days his alleged negligent conduct in dealing with the affairs of the Society as revealed in the report dated 4th October, 1969. In the meantime, P2 came to be appointed by the Registrar vide his order dated 26th July, 1969, to make inquiries under Section S of A1. Petitioner by his reply dated 18/22th January, 1970 submitted his explanation and also challenged the legality of the inquiry and the findings recorded therein. On 5th March, 1970, P3, treasurer of the Society and an employee of the Bank criminal complaints in the Court of Addl. Chief Presidency Magistrate  alleging that the appellant and two other office bearers of the society had dishonestly misappropriated a sum of Rs. 51,000/ and Rs. 80,000/- respectively which was entrusted to the appellant in his capacity as Promoter and Secretary of the Society and thereby committed criminal breach of trust. The Magistrate framed the charges against the appellant. The Bank having regard to the serious misconduct of the appellant involving moral turpitude vide its order dated 3rd November, 1970 suspended the appellant pending trial. The appellant protested this action of the Bank complaining that he was not given an opportunity of hearing before passing the order of suspension. In the meantime, P2, the authorized officer appointed by the Registrar vide his order dated 9th October, 1971 held the appellant liable to pay Rs. 2,36,000/- to the Society in addition to the amount of Rs. 2,03,000/- for which he (the appellant) and two other office bearers of the Society were held jointly liable. The Bank in view of this finding, vide its order dated 29th November, 1971 terminated the services of the appellant with effect from 1st December, 1971 along with notice pay. The appellant protested against the Action of the Bank and on 3rd December, 1971 filed detailed representation against the order of termination. The Bank replied to the appellant's representation and justified its action. The appellant on 28th December, 1971 submitted his reply to the Bank stating, inter alia, that the termination of his services was not simplicitor and was in violation of the principles of natural justice; that no opportunity of hearing was given to him; that the termination order attached stigma. The appellant aggrieved by the findings and order made by P2 appealed before Tribunal. In the meantime, the criminal proceedings ended in conviction vide order dated 27th March, 1972 passed by the Addl. Chief Metropolitan Magistrate. The appellant challenged the order of conviction and sentence in the High Court  and during the pendency of the said appeal, the Tribunal vide its order dated April 12, 1973 dismissed the appellant's appeal but reduced the liability by Rs. 72,000/-. On November 12, 1973, the High Court allowed the criminal appeal and acquitted the appellant. The High Court, however, in its order observed that since the services of the appellant were terminated in view of the criminal proceedings and since the appellant has been acquitted, representation, if a any, by the appellant to the Bank for reinstatement may be considered sympathetically. Taking clue from the observations made by the High Court, the appellant filed three representations, the last being dated 3rd May, 1975 requesting the Bank to revoke the order of termination and be reinstated. The Bank vide its communication dated May 21, 1975 refused to reinstate the appellant. The appellant, therefore, on July 23, 1975 filed the writ petition in the High Court for quashing the orders dated 29th November 1971, 27th December, 1971 and 21st May, 1975 passed by the Bank. The learned Single Judge of the High Court by his judgment and order dated December 6/7, 1979 granted desired relief to the appellant. The Bank aggrieved by the judgment and order passed by the learned Single Judge preferred an appeal under Clause 15 of the Letters Patent. The appeal was heard by the Division Bench. The Division Bench of the High Court did not agree with the judgment passed by the learned Single Judge and consequently by its judgment and order dated April 16 1985 allowed the appeal and dismissed the writ petition the ground of laches and also on merits. It is this judgment and order of the High Court which is impugned in this appeal." 
    top_documents_json = run1(query)
    # Specify the file path for the JSON file
    file_path = "data/output.json"

    # Dump the Python dictionary into a JSON file
    with open(file_path, "w") as json_file:
        json.dump(top_documents_json, json_file)