import warnings

import nltk
from ftfy import fix_text
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("portuguese")
STOPWORDS_SET = set(nltk.corpus.stopwords.words("portuguese"))


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


def get_document_metadata_paragraphs(row):

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


def get_document_metadata_entire_text(row):
    """
    Calculates metadata for the ENTIRE document, not individual paragraphs.
    """
    list_docs = []
    list_words = []
    list_stopwords = []
    list_average = []

    for doc in row["text"]:
        # Do NOT split by "\n".
        # We treat the full text as one unit for the 8192 context.
        words = doc.split()

        #  Skip if doc is too short (e.g. < 50 words) to save compute
        num_words = len(words)
        if num_words < 100:
            continue

        stopwords_cnt = 0
        total_chars = 0

        for word in words:
            total_chars += len(word)
            # Fast lookup using set
            if word.casefold() in STOPWORDS_SET:
                stopwords_cnt += 1

        avg_word_len = total_chars / num_words if num_words > 0 else 0

        list_docs.append(doc)
        list_words.append(num_words)
        list_stopwords.append(stopwords_cnt)
        list_average.append(avg_word_len)

    return {
        "text": list_docs,
        "num_words": list_words,
        "stopwords": list_stopwords,
        "average": list_average,
    }
