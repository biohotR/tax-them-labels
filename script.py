import ast

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from unidecode import unidecode

nltk.download('wordnet')
nltk.download('omw-1.4')

GENERIC_WORDS = {
    "services", "management", "solutions", "group", "and",
    "company", "system", "business", "operations", "service"
}

SYNONYM_CACHE = {}

companies_df = pd.read_csv('company_list.csv')
taxonomy_df = pd.read_csv('insurance_taxonomy.csv')

companies_df["business_tags"] = companies_df["business_tags"].apply(ast.literal_eval)

def get_synonyms(word, max_synonyms=3):
    synonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonym = normalize(lemma.name().replace("_", " "))
            if synonym != word:
                synonyms.add(synonym)
                if len(synonyms) >= max_synonyms:
                    return synonyms
    return synonyms

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = unidecode(text)
    text = text.lower().strip()
    text = ' '.join(text.split())

    return text

def flatten_tags(tags):
    if isinstance(tags, str):
        return " ".join(normalize(tag) for tag in tags)
    return ""

def score_rule_match(row, label_keywords):
    scores = {}

    description = normalize(row["description"])
    sector = normalize(row["sector"])
    category = normalize(row["category"])
    niche = normalize(row["niche"])
    tags = row["business_tags"]
    if not isinstance(tags, list):
        tags = []

    for label, keywords in label_keywords.items():
        score = 0
        desc_hits = 0

        matched_keywords = set()
        base_keywords = keywords["base"]
        synonyms = keywords["synonyms"]

        for kw in base_keywords:
            if kw in GENERIC_WORDS:
                continue
            if kw in description:
                desc_hits += 1
                matched_keywords.add(kw)
                score += 1
            if kw in sector:
                score += 1
                matched_keywords.add(kw)
            if kw in category:
                score += 2
                matched_keywords.add(kw)
            if kw in niche:
                score += 3
                matched_keywords.add(kw)
            if any(kw in normalize(tag) for tag in tags):
                score += 3
                matched_keywords.add(kw)

        for kw in synonyms:
            if kw in GENERIC_WORDS:
                continue
            if kw in description:
                score += 0.5
                matched_keywords.add(kw)
            if kw in sector:
                score += 0.5
                matched_keywords.add(kw)
            if kw in category:
                score += 1
                matched_keywords.add(kw)
            if kw in niche:
                score += 1.5
                matched_keywords.add(kw)
            if any(kw in normalize(tag) for tag in tags):
                score += 1.5
                matched_keywords.add(kw)
        
        if len(matched_keywords) < 2:
            continue

        if desc_hits >= 2:
            score += 2
        elif desc_hits == 1:
            score += 1

        if score >= 7:
            scores[label] = (score, sorted(matched_keywords))


    # return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted([(label, s_kw[0], s_kw[1]) for label, s_kw in scores.items()], key=lambda x: x[1], reverse=True)

def main():
    # normalize the taxonomy labels
    taxonomy_df["normalized_label"] = taxonomy_df["label"].apply(normalize)

    label_keywords = {}

    for _, row in taxonomy_df.iterrows():
        label = row["label"]
        base_keywords = row["normalized_label"].split()

        # expanded_keywords = set(base_keywords)
        base_set = set()
        synonym_set = set()

        for kw in base_keywords:
            if kw in GENERIC_WORDS:
                continue

            base_set.add(kw)

            if kw not in SYNONYM_CACHE:
                SYNONYM_CACHE[kw] = get_synonyms(kw)
            synonym_set.update(SYNONYM_CACHE[kw])

        label_keywords[label] = {
            "base": base_set,
            "synonyms": synonym_set
        }

    # create a new column with all the text data normalized/flattened
    companies_df["company_text"] = (
        companies_df["description"].apply(normalize) +
        companies_df["business_tags"].apply(flatten_tags) +
        companies_df["sector"].apply(normalize) + 
        companies_df["category"].apply(normalize) + 
        companies_df["niche"].apply(normalize)
    )

    companies_df["scored_rule_matches"] = companies_df.apply(
        lambda row: score_rule_match(row, label_keywords),
        axis=1
    )

    # print(companies_df[["company_text", "rule_based_matches"]].head(1))
    pd.set_option("display.max_colwidth", None)

    # print(companies_df.columns.to_list())
    print(companies_df[["company_text"]].iloc[0])
    print(companies_df[["scored_rule_matches"]].iloc[0])


    for label, score, keywords in reversed(companies_df["scored_rule_matches"].iloc[0]):
        print(f"{score:>2}  -  {label} (matched: {', '.join(keywords)})")

if __name__ == "__main__":
    main()
