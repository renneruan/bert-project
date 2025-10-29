import warnings

import nltk
from ftfy import fix_text
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("portuguese")


def paragraph_to_document(example):
    list_doc = []

    for row in example["text"]:
        str_doc = ""
        for paragraph in row["paragraphs"]:
            for doc in paragraph:
                soup = BeautifulSoup(doc, "html.parser")

                str_doc += fix_text(soup.get_text()) + "\n"

        list_doc.append(str_doc)
    return {"text": list_doc}


def get_text_metadata(row):

    list_paragraph = []
    list_words = []
    list_stopwords = []
    list_average = []

    for doc in row["text"]:
        for paragraph in doc.split("\n"):

            # strip whitespaces
            paragraph = paragraph.strip()

            # skip single or empty worded paragraphs
            if len(paragraph.split()) < 2:
                continue

            # count how many stopwords are in the paragraph
            stopwords_cnt = 0
            for word in paragraph.split():
                for stop in stopwords:
                    if stop.casefold() == word.casefold():  # insensitive case
                        stopwords_cnt += 1
                        break  # count once and speed up everything

            # count non whitespace characters
            characters = 0
            for word in paragraph.split():
                characters += len(word)

            list_paragraph.append(paragraph)
            list_words.append(len(paragraph.split()))
            list_stopwords.append(stopwords_cnt)
            list_average.append(characters / len(paragraph.split()))

    return {
        "paragraphs": list_paragraph,
        "num_words": list_words,
        "stopwords": list_stopwords,
        "average": list_average,
    }
