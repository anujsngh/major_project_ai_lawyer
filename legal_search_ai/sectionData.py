from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Reading the excel file containing the section number, description and penalty
articlesDf = pd.read_excel("data/statute_docs/Sections description.xlsx")
sectionNum = articlesDf["Section/Sub-Section"]
sectionDesc = articlesDf["Description"]
sectionPenalty = articlesDf["Penalty"]

# Sorting the similarity scores
def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))

    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

# Create the bot response
def bot_response(user_input):
    sentence_list = list(sectionDesc)
    sentence_list.append(user_input)
    count_matrix = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(count_matrix[-1], count_matrix)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    response_flag = 0

    topN = 0 
    matching_sentences = []
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            matching_sentences.append(sentence_list[index[i]])
            response_flag = 1
            topN += 1
        if topN > 3:            # For selecting top N matching sentences 
            break

    if response_flag == 0:
        return -1

    matching_sentences_indices = []
    for i in matching_sentences:
        matching_sentences_indices.append(sentence_list.index(i))

    return matching_sentences_indices

# To get the section number, description and penalty of the matching sentences
def matchingSentencesSectionNum(matchingSentenceIndices):
    output = []
    for i in matchingSentenceIndices:
        output.append([sectionNum[i], sectionDesc[i], sectionPenalty[i]])
    return output

# To get the section number, description and penalty if the input is a section description
def queryDescToNum(inputQuery):
    matchingSentenceIndices = bot_response(inputQuery)
    if matchingSentenceIndices == -1:
        output = "Please reframe your query and try again !!"
    else:
        output = matchingSentencesSectionNum(matchingSentenceIndices)
    return output

# To get the section number, description and penalty if the input is a section number
def queryNumToDesc(inputQuery):
    try:
        sectionNumIndex = list(sectionNum).index(inputQuery.upper())
    except:
        return "Enter a valid section number and try again !!"
    return [inputQuery.upper(), sectionDesc[sectionNumIndex], sectionPenalty[sectionNumIndex]]


# Driver code

def rel_statutes_search1(query=None):
    if(query):
        inputQuery = query

        if len(inputQuery) > 3:
            output = queryDescToNum(inputQuery)
        else:
            output = queryNumToDesc(inputQuery)
            
        return output
    return ""


if __name__ == "__main__":
    pass