import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

with app.setup:
    # Import some libs
    import altair as alt
    import glob
    import hashlib
    import mailparser
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import re
    import re2
    import seaborn as sns
    import spacy
    import spacy_transformers
    import tarfile
    import typing
    import wget
    import pandas as pd

    from dateutil import parser
    from bs4 import BeautifulSoup
    from collections import Counter
    from math import sqrt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from tqdm.auto import tqdm


    # Load NLP module
    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("merge_entities", after="ner")

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "TIME", "pattern": [{"SHAPE": "dd:dd:dd"}]},
    ]
    ruler.add_patterns(patterns)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data acquisition & preprocessing
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Download dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define dataset source
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_source = {
        'easy_ham': {
            'url': "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
            'count': 2500,
            'is_spam': False
        },
        'easy_ham_2': {
            'url': "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
            'count': 1400,
            'is_spam': False
        },
        'hard_ham': {
            'url': "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
            'count': 250,
            'is_spam': False
        },
        'spam': {
            'url': "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
            'count': 500,
            'is_spam': True
        },
        'spam_2': {
            'url': "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
            'count': 1397,
            'is_spam': True
        }
    }
    return (dataset_source,)


@app.cell(hide_code=True)
def _():
    dataset_dir = "./datasets/"


    def download_dataset(dataset_path: str, dataset_url: str):
        tmp = wget.download(dataset_url)

        with tarfile.open(tmp, "r:bz2") as tar:
            tar.extractall(dataset_dir)

        os.remove(os.path.join(dataset_path, "cmds"))
        os.remove(tmp)
    return dataset_dir, download_dataset


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Check dataset integrity
    """)
    return


@app.cell(hide_code=True)
def _(dataset_dir, dataset_source, download_dataset):
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)


    for _dataset_name, _dataset_info in dataset_source.items():
        _dataset_path = os.path.join(dataset_dir, _dataset_name)

        if os.path.exists(_dataset_path):

            for _filename in os.listdir(_dataset_path):
                _file_path = os.path.join(_dataset_path, _filename)

                if os.path.isfile(_file_path):
                    _md5_hash = hashlib.md5(open(_file_path, 'rb').read()).hexdigest()
                    _provided_hash = _filename.split(".")[1]

                    # Ignore wrong hash from source
                    if (_md5_hash != _provided_hash and _provided_hash != "244a63cd74c81123ef26129453e32c95"):
                        mo.md(f"File {_filename} is corrupted, redownloading dataset {_dataset_name}...")

                        download_dataset(_dataset_name, _dataset_info['url'])
                        os.remove(os.path.join(_dataset_path, _filename))

                        break

        else:
            mo.md(f"Dataset {_dataset_name} not found, downloading...")
            download_dataset(_dataset_path, _dataset_info['url'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Viewing some emails
    """)
    return


@app.cell(hide_code=True)
def _():
    mails = glob.glob("./datasets/*/*", recursive=True)
    idx = mo.ui.number(start=0, stop=len(mails) - 1, label="Number")
    return idx, mails


@app.cell(hide_code=True)
def _(idx):
    idx
    return


@app.cell(hide_code=True)
def _(idx, mails):
    mail = mailparser.parse_from_file(mails[idx.value])


    mo.md(f"""
    Path: `{mails[idx.value]}`

    **{mail.subject}**

    ---

    {mail.text_plain}

    ---

    {mail.text_html}

    ---

    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load whole dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_checkpoints = {
        'orig': {
            'description': "Dataset without preprocessing (6046 entries)",
            'checksum': "6ab4e23a696aeac72ad3b38396666a25"
        },
        'pre_cleaned': {
            'description': "Dataset after pre-cleanup (5851 entries)",
            'checksum': "48be76e083002c007728e2ccbf855e6e"
        },
        'post_cleaned': {
            'description': "Dataset after post-cleanup (5851 entries)",
            'checksum': "aeee585d4a832eee0ce9dace5c2f8ac1"
        }
    }
    return (dataset_checkpoints,)


@app.cell(hide_code=True)
def _(dataset_checkpoints, dataset_dir):
    def load_dataset(checkpoint: str) -> pd.DataFrame:
        dataset_path = os.path.join(dataset_dir, f"{checkpoint}.gzip")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No checkpoint '{checkpoint}' in {dataset_dir}")

        hash = hashlib.md5(open(dataset_path, 'rb').read()).hexdigest()
        if hash != dataset_checkpoints[checkpoint]['checksum']:
            raise ValueError("Data corrupted, or outdated. Check checksum.")

        data = pd.read_parquet(dataset_path)
        return data
    return (load_dataset,)


@app.cell(hide_code=True)
def _(dataset_dir, dataset_source, load_dataset, save_dataset):
    try:
        df = load_dataset('orig')

    except (FileNotFoundError, ValueError) as e:
        data = {
            'subject': [],
            'text': [],
            'html': [],
            'label': []
        }


        for _dataset_name, _dataset_info in dataset_source.items():
            _dataset_path = os.path.join(dataset_dir, _dataset_name)

            if os.path.exists(_dataset_path):

                for _filename in os.listdir(_dataset_path):
                    _file_path = os.path.join(_dataset_path, _filename)

                    if os.path.isfile(_file_path):
                        try:
                            _mail = mailparser.parse_from_file(_file_path)
                        except Exception as e:
                            print(f"Error with file {_file_path}: {e}")
                            continue

                        data['subject'].append(_mail.subject)
                        data['text'].append("\n".join(_mail.text_plain))
                        data['html'].append("\n".join(_mail.text_html))
                        data['label'].append(int(_dataset_info['is_spam']))

        mo.md(f"{str(e)}\nLoading from files...")

        df = pd.DataFrame(data)
        save_dataset(df, 'orig')
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Save dataset
    """)
    return


@app.cell(hide_code=True)
def _(dataset_dir):
    def save_dataset(df: pd.DataFrame, checkpoint_name: str):
        path = os.path.join(dataset_dir, f"{checkpoint_name}.gzip")
        df.to_parquet(
            path=path,
            compression='gzip'
        )

        hash = hashlib.md5(open(path, 'rb').read()).hexdigest()
        print(f"File: {path}\nMD5: {hash}")
    return (save_dataset,)


@app.cell(hide_code=True)
def _():
    hashlib.md5(open("./datasets/orig.gzip", 'rb').read()).hexdigest()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preprocess email

    1. Remove duplicates from raw dataset
    2. Find and replace certain tokens with regex (email, addresses, phone number, etc.)
    3. Find and replace proper noun tokens
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Data exploration notes

    - High correlation: HTML usage and spam
    - Dataset artifact: a lot of spam emails have this:

    ```text
    --DeathToSpamDeathToSpamDeathToSpam--
    ```

    - There might be (mostly in spam emails) forms with `_____` blanks.
    - Mailing list footer should be removed, they mostly appear in ham emails.

    ```text
    -------------------------------------------------------
    This sf.net email is sponsored by:ThinkGeek
    Welcome to geek heaven.
    http://thinkgeek.com/sf
    _______________________________________________
    Spamassassin-talk mailing list
    Spamassassin-talk@lists.sourceforge.net
    https://lists.sourceforge.net/lists/listinfo/spamassassin-talk
    ```

    - ~~Email replies appear with `> ` or `--] `, and "On DATE, USER wrote:"~~ Nvm, it's pretty random.
    - Mailling list (or some name) in subject that is not filtered (like ilug)
    - Entity labels exclusion: "CARDINAL", "ORDINAL"
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Remove duplicates
    """)
    return


@app.cell
def _(df):
    deduplicated_df = df.drop_duplicates(subset=['text', 'html'])
    return (deduplicated_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Preprocess preview
    """)
    return


@app.cell
def _(idx):
    idx
    return


@app.cell(hide_code=True)
def _(
    artifact_cleanup,
    deduplicated_df,
    html_cleanup,
    idx,
    random_cleanup,
    subject_cleanup,
):
    _tfidf = TfidfVectorizer()
    _entry = deduplicated_df.iloc[idx.value]

    _clean_subj = subject_cleanup(_entry['subject'])
    _clean_html = html_cleanup(_entry['html'])
    _merged_text = merge_text_html(_entry['text'],_entry['html'])
    _text = f"{_clean_subj}\n{_merged_text}"

    _text_2 = random_cleanup(_text)
    _text_3 = artifact_cleanup(_text_2)

    _doc = nlp(_text_3)
    _newtext = nlp_pipeline(_doc)


    _table2 = mo.ui.table(
        data=[{
            'Token': token.text,
            'Lemma': token.lemma_,
            'PoS': token.pos_,
            'Tag': token.tag_,
            'Tag (explain)': spacy.explain(token.tag_),
            'Entity': token.ent_type_,
            'Stop word': token.is_stop,
            'URL': token.like_url,
            'Email': token.like_email
        } for token in _doc],
        pagination=True
    )

    _table3 = mo.ui.table(
        data=[{
            'Entity': ent.text,
            'Label': ent.label_
        } for ent in _doc.ents],
        pagination=True
    )


    if len(_newtext) == 0:
        _newtext = "_<EMPTY>_"

    _results = _tfidf.fit_transform([_newtext])
    _vocab = [{'word': _word, 'idx': _idx} for _word, _idx in _tfidf.vocabulary_.items()]
    _table = mo.ui.table(data=_vocab, pagination=True)


    mo.vstack(
        [
            mo.md(f"Label: {"Ham" if _entry['label'] == 0 else "Spam"}"),
            mo.hstack(
                [
                    mo.md(f"{_clean_subj}\n{_merged_text}".replace("\n", " ")),
                    mo.md(_text_3.replace("\n", " ")),
                    mo.md(_newtext.replace("\n", " "))
                ],
                widths="equal"
            ),
            mo.md(f"Stop word ratio: {len([tok for tok in _doc if tok.is_stop]) / _doc.__len__()}"),
            mo.hstack([
                _table,
                _table2,
                _table3
            ])
        ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pre-cleanup
    """)
    return


@app.cell(hide_code=True)
def _():
    # Remove some "---", "###", "***", etc.
    unamed_pattern = re2.compile(
        r"[-#*_=+]{3,}"
    )

    url_pattern = re.compile(
        r"((http|ftp|https):\/\/)?([\w_-]+(?:\.[\w_-]+)*\.[a-zA-Z_-][\w_-]+)([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    )

    email_pattern = re.compile(
        r"<[^>]+@[^>]+>"
    )

    # Remove mailing list information
    subject_cleanup_pattern = re2.compile(
        r"\[.*?\]",
    )


    def artifact_cleanup(text):
        artifact = "--DeathToSpamDeathToSpamDeathToSpam--"
        return text.replace(artifact, "")


    def random_cleanup(text: str) -> str:
        return re2.sub(
            pattern=unamed_pattern,
            text=text,
            repl=""
        )


    def subject_cleanup(text: str) -> str:
        return re2.sub(
            pattern=subject_cleanup_pattern,
            text=text,
            repl=""
        ).strip()


    def html_cleanup(html):
        if pd.isna(html):
            return ""

        soup = BeautifulSoup(html, 'html5lib')

        return soup.get_text(separator=" ", strip=True)
    return (
        artifact_cleanup,
        html_cleanup,
        random_cleanup,
        subject_cleanup,
        url_pattern,
    )


@app.function(hide_code=True)
def cosine_similarity(s1: str, s2: str) -> float:
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    # Convert strings to character frequency vectors
    vec1 = Counter(s1)
    vec2 = Counter(s2)

    # Calculating cosine similarity
    dot_product = sum(vec1[ch] * vec2[ch] for ch in vec1)
    magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
    magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))
    return dot_product / (magnitude1 * magnitude2)


@app.cell(hide_code=True)
def _(url_pattern):
    def _merge_text_html(text: str, html: str, threshold = 0.95):
        merged_text = {
            'str': "",
            'link_count': len(url_pattern.findall(text))
        }

        if len(html) != 0:
            soup = BeautifulSoup(html, 'html5lib')
            clean_html = soup.get_text(separator=" ", strip=True)

            if len(text) == 0:
                merged_text['str'] = clean_html.replace("\n", " ")
                merged_text['link_count'] += len(soup.find_all('a'))

            else:
                text_no_url = url_pattern.sub(string=text, repl="")
                similarity = cosine_similarity(text_no_url, clean_html)
                # print(similarity)

                if similarity >= threshold:
                    merged_text['str'] = text_no_url

                else:
                    merged_text['str'] = f"{text_no_url}\n{clean_html}"
                    merged_text['link_count'] += len(soup.find_all('a'))

        else:
            merged_text['str'] = url_pattern.sub(string=text, repl="")

        return merged_text
    return


@app.function(hide_code=True)
def merge_text_html(text: str, html: str, threshold = 0.95) -> str:
    if len(html) != 0:
        soup = BeautifulSoup(html, 'html5lib')
        clean_html = soup.get_text(separator=" ", strip=True)

        # Artificially add _URL_ tokens
        clean_html += " google.com" * len(soup.find_all('a'))

        if len(text) == 0:
            return clean_html

        else:
            similarity = cosine_similarity(text, clean_html)

            if similarity >= threshold:
                return text

            else:
                return f"{text}\n{clean_html}"

    else:
        return text


@app.cell(hide_code=True)
def _(artifact_cleanup, random_cleanup, subject_cleanup):
    def pre_cleanup(subject: str, text: str, html: str):
        clean_subj = subject_cleanup(subject)
        merged_text = merge_text_html(text, html)

        pass_1 = f"{clean_subj}\n{merged_text}"
        pass_2 = random_cleanup(pass_1)
        pass_3 = artifact_cleanup(pass_2)

        return pass_3
    return (pre_cleanup,)


@app.cell(hide_code=True)
def _(deduplicated_df, load_dataset, pre_cleanup, save_dataset):
    try:
        pre_cleaned_df = load_dataset('pre_cleaned')

    except (FileNotFoundError, ValueError) as e:
        tqdm.pandas(desc="Running pre_cleanup", ncols=100)

        pre_cleaned_df = pd.DataFrame({
            'text': deduplicated_df.progress_apply(
                lambda x: pre_cleanup(x['subject'], x['text'], x['html']),
                axis=1
            ),
            'label': deduplicated_df['label']
        })

        save_dataset(pre_cleaned_df, 'pre_cleaned')
    return (pre_cleaned_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Post cleanup
    """)
    return


@app.cell
def _(load_dataset, pre_cleaned_df, save_dataset):
    try:
        post_cleaned_df = load_dataset('post_cleaned')

    except (FileNotFoundError, ValueError) as e:
        _post_cleaned_emails = []

        _post_cleanup_status = mo.status.progress_bar(
            title="Running post_cleanup",
            total=len(pre_cleaned_df)
        )

        with _post_cleanup_status as _bar:
            for _doc in nlp.pipe(
                texts=pre_cleaned_df['text'].tolist(),
                disable=["parser"]
            ):
                _post_cleaned_emails.append(nlp_pipeline(_doc))

                _bar.update()

        post_cleaned_df = pd.DataFrame({
            'text': _post_cleaned_emails,
            'label': pre_cleaned_df['label']
        })

        pre_cleaned_df[post_cleaned_df['text'] == ""]
        post_cleaned_df = post_cleaned_df[post_cleaned_df['text'] != ""].reset_index(drop=True)

        save_dataset(post_cleaned_df, 'post_cleaned')
    return (post_cleaned_df,)


@app.function(hide_code=True)
def token_processor(token: spacy.tokens.Token) -> str:
    if token.is_stop:
        return ""

    if token.like_url:
        return "_url_"
    if (token.like_email) or (token.text.startswith("<") and "@" in token.text and token.text.endswith(">")):
        return "_email_"

    if token.ent_type_ not in ["", "CARDINAL", "ORDINAL"]:
        return f"_{token.ent_type_}_".lower()

    if token.pos_ == "PROPN":
        return "_propn_"
    if token.pos_ == "NUM":
        return "_num_"

    if token.tag_ == "LS":
        return ""
        
    else:
        return token.lemma_


@app.function(hide_code=True)
def nlp_pipeline(doc: spacy.tokens.Doc, stop_threshold = 0.01) -> str:
    #with doc.retokenize() as retokenizer:
    #    for ent in doc.ents:
    #        if ent.label_ in ["DATE", "TIME", "MONEY"]:
    #            retokenizer.merge(ent)

    #    for match in email_pattern.finditer(doc.text):
    #        span = doc.char_span(match.start(), match.end())
    #        if span is not None:
    #            retokenizer.merge(span)

    # Treat as spam
    stop_word_ratio = len([token for token in doc if token.is_stop]) / doc.__len__()
    if stop_word_ratio < stop_threshold:
        return ""

    return " ".join(
        [token_processor(token) for token in doc]
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Features engineering
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Additional features to consider

    - ~~Contains HTML (boolean)~~
    - Number of links (share the same token with links in text)
    - ~~Number of special characters~~ Might not be a good feature
    - Capitals ratio
    - ~~Email size~~ (maybe not)
    - Reply ratio, reply depth count
    - Stop word ratio
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Capitals ratio
    """)
    return


@app.function
def calc_capitals_ratio(text: str) -> float:
    if len(text) == 0:
        return 0.0

    capitals_count = sum(1 for c in text if c.isupper())
    return capitals_count / len(text)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Number of special characters
    """)
    return


@app.function
def count_special_chars(text):
    special_chars = "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~"
    return sum(1 for char in text if char in special_chars)


@app.cell
def _(df):
    df['text'].apply(count_special_chars)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### TF-IDF
    """)
    return


@app.cell
def _(post_cleaned_df):
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the data and transform it into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(post_cleaned_df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return (tfidf_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train time
    """)
    return


@app.cell
def _(post_cleaned_df, tfidf_df):
    X = tfidf_df
    y = post_cleaned_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    return logreg, y_pred, y_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Confusion time
    """)
    return


@app.cell
def _(y_pred, y_test):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    return


@app.cell
def _(y_pred, y_test):
    target_names = ["Ham", "Spam"]
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model analysis
    """)
    return


@app.cell
def _(logreg, tfidf_df):
    features_df = pd.DataFrame({
        'word': tfidf_df.columns,
        'weight': logreg.coef_[0]
    })


    top_spam = features_df.nlargest(20, 'weight')
    top_ham = features_df.nsmallest(20, 'weight')


    plot_df = pd.concat([
        top_spam.assign(type='Spam'),
        top_ham.assign(type='Ham')
    ])

    plot_df['type'] = plot_df['weight'].apply(lambda x: 'Spam' if x > 0 else 'Ham')
    return top_ham, top_spam


@app.cell
def _(top_ham, top_spam):
    mo.hstack(
        [
            mo.ui.altair_chart(
                alt.Chart(top_ham).mark_bar().encode(
                    x='weight:Q',
                    y=alt.Y('word:N', sort='null'),
                    color=alt.Color('weight:Q', scale=alt.Scale(scheme='greens', reverse=True, type='log')),
                    tooltip=['word', 'weight']
                ).properties(
                    title="Ham words",
                    height=600
                )
            ),
            mo.ui.altair_chart(
                alt.Chart(top_spam).mark_bar().encode(
                    x='weight:Q',
                    y=alt.Y('word:N', sort='null'),
                    color=alt.Color('weight:Q', scale=alt.Scale(scheme='reds', type='log')),
                    tooltip=['word', 'weight']
                ).properties(
                    title="Spam words",
                    height=600
                )
            )
        ],
        justify='center',
        widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Random stuff
    """)
    return


@app.cell
def _(pre_cleaned_df):
    mo.ui.dataframe(pre_cleaned_df)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Beautiful Soup playground
    """)
    return


@app.cell(hide_code=True)
def _():
    html_input = mo.ui.text_area(placeholder="Paste HTML code...", full_width=True)
    html_input 
    return (html_input,)


@app.cell(hide_code=True)
def _(html_input):
    soup = BeautifulSoup(html_input.value, 'html5lib')
    cleaned_html = soup.get_text(separator=" ", strip=True)
    print(cleaned_html)
    return


@app.cell
def _(count_links, html_input):
    print(count_links(html_input.value))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## RE2 playground
    """)
    return


@app.cell
def _():
    text_input = mo.ui.text_area(placeholder="Paste text...", full_width=True)
    text_input
    return (text_input,)


@app.cell
def _():
    regex_input = mo.ui.text(placeholder="Regex", full_width=True)
    regex_input
    return (regex_input,)


@app.cell
def _(regex_input, text_input):
    re2.sub(pattern=regex_input.value, text=text_input.value, repl="")
    return


if __name__ == "__main__":
    app.run()
