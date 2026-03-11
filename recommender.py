import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(series):
    return series.fillna("").str.lower().str.strip()


def compact_tokens(series):
    return (
        series.fillna("")
        .apply(lambda x: " ".join(t.strip().replace(" ", "") for t in x.split(",")))
        .str.lower()
    )


class MovieRecommender:
    def __init__(self, csv_path="data_ass2_part2_wk5.csv"):
        self.df = pd.read_csv(csv_path)

        self.df["feat_description"] = clean_text(self.df["description"])
        self.df["feat_genres"] = compact_tokens(self.df["listed_in"])
        self.df["feat_director"] = compact_tokens(self.df["director"])
        self.df["feat_cast"] = compact_tokens(self.df["cast"])

        # Build weighted feature soup.
        self.df["soup"] = (
            self.df["feat_genres"]
            + " "
            + self.df["feat_genres"]
            + " "
            + self.df["feat_director"]
            + " "
            + self.df["feat_director"]
            + " "
            + self.df["feat_cast"]
            + " "
            + self.df["feat_description"]
        )

        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
            max_features=12000,
        )

        self.tfidf = tfidf
        self.tfidf_matrix = tfidf.fit_transform(self.df["soup"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df["title"].str.lower()).drop_duplicates()

    def get_recommendations(self, title, n=10):
        key = title.strip().lower()

        if key not in self.indices:
            matches = sorted([t for t in self.indices.index if key in t])
            if not matches:
                return None, None
            key = matches[0]

        idx = self.indices[key]
        sim_scores = sorted(
            enumerate(self.cosine_sim[idx]), key=lambda x: x[1], reverse=True
        )[1 : n + 1]

        rec_idx = [i for i, _ in sim_scores]
        scores = [round(s * 100, 2) for _, s in sim_scores]

        result = self.df.iloc[rec_idx][["title", "listed_in", "director"]].copy()
        result.columns = ["title", "genres", "director"]
        result.insert(0, "rank", range(1, n + 1))
        result["similarity_score (%)"] = scores
        result.reset_index(drop=True, inplace=True)

        return key, result