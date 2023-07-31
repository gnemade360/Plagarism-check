import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

scholar_files = [docu for docu in os.listdir() if docu.endswith('.txt')]
scholar_notes = [open(_file, encoding='utf-8').read()
                 for _file in scholar_files]


def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(docu1, docu2): return cosine_similarity([docu1, docu2])


vectors = vectorize(scholar_notes)
s_vectors = list(zip(scholar_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results


for data in check_plagiarism():
    print(data)
