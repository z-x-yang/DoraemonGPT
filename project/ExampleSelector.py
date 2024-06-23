import json
import pdb

from langchain.prompts.example_selector.base import BaseExampleSelector
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer


class CustomExampleSelector(BaseExampleSelector):
    def __init__(self):
        with open("./project/examples.json", "r") as file:
            data = json.load(file)
        # print("load examples:",data)
        self.examples = data

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: str) -> List[dict]:
        """Select which examples to use based on the inputs."""
        max_similarity = -1
        most_similar_example = None

        # 预处理字符串文本
        # str_tokens = self.preprocess_text(input_variables)
        str_tokens = input_variables

        # 创建TF-IDF向量化器
        vectorizer = TfidfVectorizer()

        for item in self.examples:
            # 获取字典中指定键对应的值
            dict_value = item.get("Input")
            if dict_value is None:
                raise KeyError("Key not found in the dictionary.")

            # 预处理字典值文本
            # dict_tokens = self.preprocess_text(dict_value)
            dict_tokens = dict_value

            # 将预处理后的文本转换为TF-IDF向量表示
            vectors = vectorizer.fit_transform([str_tokens, dict_tokens])
            # 计算余弦相似度
            similarity = cosine_similarity(vectors[0], vectors[1])

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_example = item
        # 返回最相似的值
        return most_similar_example

    # def preprocess_text(self, text):
    #     # 分词
    #     tokens = word_tokenize(text)
    #     # 去除停用词
    #     stop_words = set(stopwords.words('english'))
    #     filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #     # 词形还原
    #     lemmatizer = WordNetLemmatizer()
    #     lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    #     # 返回处理后的文本
    #     return lemmas
