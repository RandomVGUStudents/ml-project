import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="full",
    app_title="Spam Classifier",
    auto_download=["ipynb", "html"],
)

with app.setup(hide_code=True):
    import marimo as mo

    # Miscellaneous
    import glob
    import hashlib
    import os
    import tarfile
    import typing
    import wget
    from collections import Counter
    from math import sqrt
    from tqdm.auto import tqdm

    # Data processing
    import numpy as np
    import re
    import re2
    import spacy
    import spacy_transformers
    import pandas as pd
    from bs4 import BeautifulSoup
    import mailparser

    # Plotting
    import altair as alt
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Scikit-learn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics


    # Load NLP module
    print(spacy.prefer_gpu())
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
    # Hi there!

    This is the notebook for our Machine Learning project.
    <br>
    Let's take a look at the table of content (it is also available on the right side of the screen):

    I. Data Acquisition
    - Download dataset
    - Load whole dataset
    - Preprocess email

    II. Feature Engineering
    - Additional features to consider

    III. Training
    - Train time
    - Confusion time

    IV. Model Analysis

    **Let's get started!**

    💡 Tips: Click ▶️ in the down right corner.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # I. Data Acquisition

    In this section, we obtain the data from the Apache's SpamAssassin Public Corpus and read them.
    <br>
    After that, we can load all emails into a DataFrame.

    ## 1. Download dataset
    """)
    return


@app.cell
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
            'count': 1396,
            'is_spam': True
        }
    }

    dataset_dir = "./datasets/"

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)


    def download_dataset(dataset_path: str, dataset_url: str):
        tmp = wget.download(dataset_url)

        with tarfile.open(tmp, "r:bz2") as tar:
            tar.extractall(dataset_dir)

        os.remove(os.path.join(dataset_path, "cmds"))
        os.remove(tmp)


    mo.md(r"""
    ### a. Define dataset source and download function

    We have 5 files:

    - easy_ham (2500 ham emails)
    - easy_ham_2 (1400 ham emails)
    - hard_ham (250 ham emails)
    - spam (500 spam emails)
    - spam_2 (1396 spam emails) *Although they said there were 1397 emails, there are only 1396 files in the extracted folder.*

    Our download function will download the files and extract into `./datasets/` folder.
    """)
    return dataset_dir, dataset_source, download_dataset


@app.cell
def _(dataset_dir, dataset_source, download_dataset):
    for _dataset_name, _dataset_info in dataset_source.items():
        _dataset_path = os.path.join(dataset_dir, _dataset_name)

        if os.path.exists(_dataset_path):

            if len(os.listdir(_dataset_path)) != _dataset_info['count']:
                print(f"Dataset {_dataset_name} is missing some files, let's redownload.")
                download_dataset(_dataset_name, _dataset_info['url'])

            for _filename in os.listdir(_dataset_path):
                _file_path = os.path.join(_dataset_path, _filename)

                if os.path.isfile(_file_path):
                    _md5_hash = hashlib.md5(open(_file_path, 'rb').read()).hexdigest()
                    _provided_hash = _filename.split(".")[1]

                    # Ignore wrong hash from source
                    if (_md5_hash != _provided_hash and _provided_hash != "244a63cd74c81123ef26129453e32c95"):
                        print(f"Hash mismatch for {_filename}, redownloading dataset {_dataset_name}...")

                        download_dataset(_dataset_name, _dataset_info['url'])
                        os.remove(os.path.join(_dataset_path, _filename))

                        break

        else:
            print(f"Missing dataset {_dataset_name}, downloading it now...")
            download_dataset(_dataset_path, _dataset_info['url'])


    mo.md(r"""
    ### b. Check dataset integrity

    The file name of each email is its MD5 hash.
    <br>
    We can use it to verify the integrity of the dataset.

    *Note: A file has incorrect hash from source, that's not our fault.*

    We will redownload the dataset if any file is corrupted or missing.
    """)
    return


@app.cell
def _():
    mails = glob.glob("./datasets/*/*", recursive=True)
    idx = mo.ui.number(start=0, stop=len(mails) - 1, label="Number")

    mo.md(r"""
    ### c. Viewing some emails

    Let's take a look at some emails that we've got.

    To save us some hassle, we won't be looking as raw email files (feel free to look at them, though!)
    <br>
    However, we will be using the `mailparser` library to parse the emails.

    What we care about:

    - Subject
    - Plain text content
    - HTML content

    💡 Tips: Click "Fullscreen" or "Expand output" button (they will appear on the right side when you hover the box below).
    <br>
    Modify the number to view different emails.
    """)
    return idx, mails


@app.cell
def _(idx, mails):
    mail = mailparser.parse_from_file(mails[idx.value])


    mo.vstack([
        idx,
        mo.md(f"""
    Path: `{mails[idx.value]}`

    ```text
    -- SUBJ --
    {mail.subject}**
    -- TEXT --
    {"---\n\n---".join(mail.text_plain) if mail.text_plain else "<No plain text content>"}
    -- HTML --
    {mail.text_html if mail.text_html else "<No HTML content>"}
    ```
    """)
    ])
    return


@app.cell
def _(dataset_dir):
    dataset_checkpoints = {
        'orig': {
            'description': "Dataset without preprocessing (6046 entries)",
            'checksum': "6ab4e23a696aeac72ad3b38396666a25"
        },
        'pre_cleaned': {
            'description': "Dataset after pre-cleanup (5851 entries)",
            'checksum': "589f556bc02bdeaa376ca92a99e9755c"
        },
        'post_cleaned': {
            'description': "Dataset after post-cleanup (5851 entries)",
            'checksum': "d972cc17841c34ed3f73c1bd3e3fad62"
        }
    }

    def load_dataset(checkpoint: str) -> pd.DataFrame:
        dataset_path = os.path.join(dataset_dir, f"{checkpoint}.gzip")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"No checkpoint '{checkpoint}' in {dataset_dir}")

        hash = hashlib.md5(open(dataset_path, 'rb').read()).hexdigest()
        if hash != dataset_checkpoints[checkpoint]['checksum']:
            print(f"Expected: {dataset_checkpoints[checkpoint]['checksum']}, got: {hash}")
            raise ValueError("Dataset file broken.")

        data = pd.read_parquet(dataset_path)
        return data

    def save_dataset(df: pd.DataFrame, checkpoint_name: str):
        path = os.path.join(dataset_dir, f"{checkpoint_name}.gzip")
        df.to_parquet(
            path=path,
            compression='gzip'
        )

        hash = hashlib.md5(open(path, 'rb').read()).hexdigest()
        print(f"File: {path}\nMD5: {hash}")


    mo.md(r"""
    ## 2. Load whole dataset

    Now let's load everything into a DataFrame.

    Let's understand why we need checkpoints.
    <br>
    We don't want to reprocess the dataset every time we run the notebook, so we will save the processed dataset into a file.
    <br>
    In subsequent runs, we can just load it back.

    Here we have some checkpoints and some functions to save and load datasets.
    """)
    return load_dataset, save_dataset


@app.cell
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

        print(f"{str(e)}\nLoading from files...")

        df = pd.DataFrame(data)
        save_dataset(df, 'orig')


    mo.md(r"""
    We use `mailparser` to parse 6k+ emails into a DataFrame, and save it for later.

    We will only do that when the checkpoint isn't available, or is somehow broken.
    """)
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Preprocess email

    A little rant about how ML works in this problem:

    - Emails are broken into 'tokens'. Tokens are words (in simple form), or special ones.
    - The tokens are the features for the model.

    Here's what we need to do:

    1. Remove duplicates from raw dataset.
    2. Find and replace certain tokens with regex.
    3. Find and replace tokens using Natural Language Processing (NLP).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### a. Data exploration notes

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
    - Some emails are malformed, they are extremely long
    """)
    return


@app.cell
def _(df):
    deduplicated_df = df.drop_duplicates(subset=['text', 'html'])


    mo.md(r"""
    ### b. Remove duplicates
    """)
    return (deduplicated_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### c. Preprocess preview

    This is merely the scratchpad for whatever we want to do with these emails.

    Change the number to view different emails, just like above.
    <br>
    *(Also don't forget to use fullscreen or expand output)*

    The first column contains subject + plain text + parsed HTML (more information down below).
    <br>
    The second column shows how each token is transformed (in `nlp_pipeline` section down below).
    <br>
    The third one shows the results.
    """)
    return


@app.cell
def _(
    artifact_cleanup,
    deduplicated_df,
    html_cleanup,
    idx,
    merge_text_html,
    nlp_pipeline,
    subject_cleanup,
):
    _tfidf = TfidfVectorizer()
    _entry = deduplicated_df.iloc[idx.value]

    _clean_subj = subject_cleanup(_entry['subject'])
    _clean_html = html_cleanup(_entry['html'])
    _merged_text = merge_text_html(_entry['text'],_entry['html'])
    _text = f"{_clean_subj}\n{_merged_text}"

    _text_2 = artifact_cleanup(_text)

    _doc = nlp(_text_2)
    _debug_text = nlp_pipeline(_doc, debug=True)
    _new_text = nlp_pipeline(_doc)


    _table2 = mo.ui.table(
        data=[{
            "Token": token.text,
            "Lemma": token.lemma_,
            "PoS": token.pos_,
            "Tag": token.tag_,
            "Entity": token.ent_type_,
            "Stop word": token.is_stop,
            "URL": token.like_url,
            "Email": token.like_email
        } for token in _doc],
        label="Tokens",
        pagination=True,
        wrapped_columns=["Token", "Lemma"]
    )


    if len(_new_text) == 0:
        _new_text = "_<EMPTY>_"

    _results = _tfidf.fit_transform([_new_text])
    _vocab = [{'word': _word, 'idx': _idx} for _word, _idx in _tfidf.vocabulary_.items()]
    _table = mo.ui.table(data=_vocab, label="TF-IDF", pagination=True)


    mo.vstack(
        [
            idx,
            mo.md(f"Label: {"Ham" if _entry['label'] == 0 else "Spam"}"),
            mo.hstack(
                [
                    mo.md(f"{_clean_subj}\n{_merged_text}".replace("\n", " ")),
                    mo.md(_debug_text.replace("\n", " ")),
                    mo.md(_new_text.replace("\n", " "))
                ],
                widths=[1, 1.2, 0.9]
            ),
            mo.hstack([
                _table,
                _table2
            ], widths=[1, 5]),
            mo.md(f"Stop word ratio: {len([tok for tok in _doc if tok.is_stop]) / _doc.__len__()}"),
        ])
    return


@app.cell
def _(deduplicated_df, load_dataset, pre_cleanup, save_dataset):
    try:
        #raise(ValueError("Code changed. Rebuilding dataset..."))
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


    mo.md(r"""
    ### d. Pre cleanup

    Find and replace certain tokens with regex.

    For now, in some emails' subject line, there is mailing list information. E.g.: [ILUG], or [zzzteana].
    <br>
    This should be removed, otherwise the model would learn them as ham indicator.

    It's not technically wrong, but it doesn't generalize. Those mailing lists are specific to this dataset.

    Another artifact is `--DeathToSpamDeathToSpamDeathToSpam--`.
    <br>
    It is suspected to be related to the spam report campaign, where people forward spam emails to the collector, who built this dataset.
    """)
    return (pre_cleaned_df,)


@app.cell
def _(artifact_cleanup, merge_text_html, subject_cleanup):
    def pre_cleanup(subject: str, text: str, html: str):
        clean_subj = subject_cleanup(subject)
        merged_text = merge_text_html(text, html)

        pass_1 = f"{clean_subj}\n{merged_text}"
        pass_2 = artifact_cleanup(pass_1)

        return pass_2


    mo.md(r"""
    This is the `pre_cleanup` function. It will remove the mailing list information, merge plain text and HTML into a single container, and remove artifact.
    """)
    return (pre_cleanup,)


@app.cell
def _(cosine_similarity, html_cleanup):
    def merge_text_html(text: str, html: str, threshold = 0.95) -> str:
        if len(html) != 0:
            html = html_cleanup(html)

            # Some texts has leftover HTML somehow
            text = html_cleanup(text)

            if len(text) == 0:
                return html

            else:
                similarity = cosine_similarity(text, html)

                if similarity >= threshold:
                    return text

                else:
                    return f"{text}\n{html}"

        else:
            return html_cleanup(text)


    mo.md(r"""
    Some emails have either plain text or HTML content, or both.
    <br>
    Some emails have the same content in both plain text and HTML.

    If the plain text and HTML content have a [cosine similarity](https://www.geeksforgeeks.org/python/python-similarity-metrics-of-strings/) of greater than `threshold`, they will be considered to be the same.
    """)
    return (merge_text_html,)


@app.cell
def _():
    # url_pattern = re.compile(
    #     r"((http|ftp|https):\/\/)?([\w_-]+(?:\.[\w_-]+)*\.[a-zA-Z_-][\w_-]+)([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    # )

    # Remove mailing list information
    subject_cleanup_pattern = re2.compile(
        r"\[.*?\]"
    )


    def artifact_cleanup(text):
        artifact = "--DeathToSpamDeathToSpamDeathToSpam--"
        return text.replace(artifact, "")


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

        # Artificially add _URL_ tokens
        return soup.get_text(separator=" ", strip=True) + " google.com" * len(soup.find_all('a'))


    mo.md(r"""
    While parsing HTML content, links (`<a>` tags) will be lost.
    <br>
    The function adds "google.com" for each link found, which will be interpreted as an `_url_` token.
    """)
    return artifact_cleanup, html_cleanup, subject_cleanup


@app.cell
def _():
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


    mo.md(r"""
    From the GeeksForGeeks page mentioned above.
    """)
    return (cosine_similarity,)


@app.cell
def _(
    load_dataset,
    nlp_pipeline,
    pre_cleaned_df,
    process_batches,
    save_dataset,
):
    MAX_CHAR_LENGTH = 15000


    try:
        #raise(ValueError("Code changed. Rebuilding dataset..."))
        post_cleaned_df = load_dataset('post_cleaned')

    except (FileNotFoundError, ValueError) as e:
        _post_cleaned_emails = []
        _truncated_texts = []
        _skipped_idx = []

        for i, row in pre_cleaned_df.iterrows():
            if len(row['text']) > MAX_CHAR_LENGTH:
                _skipped_idx.append(i)
                _truncated_texts.append(row['text'][:MAX_CHAR_LENGTH])
            else:
                _truncated_texts.append(row['text'])

        _post_cleanup_status = mo.status.progress_bar(
            title="Running post_cleanup",
            total=len(pre_cleaned_df)
        )

        with _post_cleanup_status as _bar:
            for _doc in process_batches(
                nlp=nlp,
                texts=_truncated_texts,
                batch_size=5
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


    mo.md(r"""
    ### e. Post cleanup

    Find and replace tokens with Natural Language Processing (NLP).

    First, we need to truncate long emails. Here, a maximum of 15k characters for each email is enforced.
    """)
    return (post_cleaned_df,)


@app.cell
def _():
    import torch

    def process_batches(nlp, texts, batch_size=None):
        for i, doc in enumerate(nlp.pipe(
            texts,
            batch_size=batch_size,
            disable=["parser"]
        )):
            doc._.trf_data = None

            yield doc

            if (i % batch_size == 0) and (torch.cuda.is_available()):
                torch.cuda.empty_cache() # VRAM explosion


    mo.md(r"""
    We use spaCy library for this job. We split the texts into `batch_size`, and clean up GPU VRAM (if it's available).
    """)
    return (process_batches,)


@app.cell
def _(token_processor):
    # Email matching
    email_pattern = re2.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )

    pgp_signature_pattern = re2.compile(
        r"(?s)-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----"
    )


    def nlp_pipeline(doc: spacy.tokens.Doc, stop_threshold = 0.01, debug = False) -> str:
        with doc.retokenize() as retokenizer:
            for match in email_pattern.finditer(doc.text):
                span = doc.char_span(
                    match.start(),
                    match.end(),
                    alignment_mode='expand'
                )

                if span is not None:
                    retokenizer.merge(span)
                    for token in span:
                        token.ent_type_ = "EMAIL"

        with doc.retokenize() as retokenizer:
            for match in pgp_signature_pattern.finditer(doc.text):
                span = doc.char_span(
                    match.start(),
                    match.end(),
                    alignment_mode='expand'
                )

                if span is not None:
                    retokenizer.merge(span)
                    for token in span:
                        token.ent_type_ = "PGP_SIG"

        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                if ent.label_ in ["DATE", "TIME", "MONEY"]:
                    retokenizer.merge(ent)


        # Treat as spam
        stop_word_ratio = len([token for token in doc if token.is_stop]) / doc.__len__()
        if stop_word_ratio < stop_threshold:
            return ""

        return " ".join(
            [token_processor(token, debug) for token in doc]
        )


    mo.md(r"""
    There are two things that the spaCy transformer model struggles with: email addresses and PGP signature.
    <br>
    We give it some help using regex. Then we can set entity type for found items as "EMAIL" or "PGP_SIG".

    About [stop words](https://en.wikipedia.org/wiki/Stop_word), if there are so few in the email, it's not normal. We set a ratio threshold of 0.01.
    <br>
    In this processing step, we simply ignore such emails. In the real application, we might directly conclude the email to be spam.
    """)
    return (nlp_pipeline,)


@app.cell
def _():
    def token_processor(token: spacy.tokens.Token, debug: bool = False) -> str:
        def debug_fmt(tag: str) -> str:
            spaced_text = "~ ~".join(token.text.split())
            return f"{tag} ~<-{spaced_text}~"

        if token.is_stop or token.tag_ == "LS":
            return "" if not debug else f"~~{token.text}~~"

        if token.like_url:
            return "_url_" if not debug else debug_fmt("_url_")

        if token.like_email:
            return "_email_" if not debug else debug_fmt("_email_")

        if token.ent_type_ and token.ent_type_ not in ["CARDINAL", "ORDINAL"]:
            tag = f"_{token.ent_type_}_".lower()
            return tag if not debug else debug_fmt(tag)

        if token.pos_ == "PROPN":
            return "_propn_" if not debug else debug_fmt("_propn_")

        if token.pos_ == "NUM":
            return "_num_" if not debug else debug_fmt("_num_")

        if token.pos_ == "X" and token.tag_ == "FW":
            return "_foreign_" if not debug else debug_fmt("_foreign_")

        return token.lemma_


    mo.md(r"""
    We replace certain tokens with the desired classification. It is demonstrated in "Preprocess preview" section.
    """)
    return (token_processor,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # II. Features engineering

    ## 1. Additional features to consider

    - ~~Contains HTML (boolean)~~
    - Number of links (share the same token with links in text)
    - ~~Number of special characters~~ Might not be a good feature
    - Capitals ratio
    - ~~Email size~~ (maybe not)
    - Reply ratio, reply depth count
    - Stop word ratio
    """)
    return


@app.cell
def _():
    def calc_capitals_ratio(text: str) -> float:
        if len(text) == 0:
            return 0.0

        capitals_count = sum(1 for c in text if c.isupper())
        return capitals_count / len(text)


    mo.md(r"""
    ### a. Capitals ratio

    I didn't use this yet, though.
    """)
    return


@app.cell
def _():
    def count_special_chars(text):
        special_chars = "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~"
        return sum(1 for char in text if char in special_chars)

    #df['text'].apply(count_special_chars)


    mo.md(r"""
    ### b. Number of special characters

    On second thought..., I skipped it.
    """)
    return


@app.cell
def _(post_cleaned_df):
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the data and transform it into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(post_cleaned_df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


    mo.md(r"""
    ### c. TF-IDF

    Code copied from GeeksForGeeks:
    """)
    return (tfidf_df,)


@app.cell
def _(post_cleaned_df, tfidf_df):
    X = tfidf_df
    y = post_cleaned_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)


    mo.md(r"""
    # III. Model training

    ## 1. Train time

    Here, we train our Logistic Regression model with 75-25 train-test split.
    """)
    return logreg, y_pred, y_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Confusion time
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


@app.cell
def _():
    mo.md(r"""
    # IV. Model analysis

    We take a look into our model and see its decision-making.

    ## 1. Important words

    The charts show top 20 words for words that are likely in ham and spam emails.

    ## 2. Email test

    I want to write another section to see how each word in an email contributes to the final conclusion. Lazy I am, though.
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


@app.cell
def _():
    mo.md(r"""
    # Random stuff
    """)
    return


@app.cell
def _(pre_cleaned_df):
    mo.ui.dataframe(pre_cleaned_df)
    return


@app.cell
def _(pre_cleaned_df):
    pre_cleaned_df['text'].apply(lambda x: len(x))
    return


@app.cell
def _():
    mo.md(r"""
    ## Beautiful Soup playground
    """)
    return


@app.cell
def _():
    html_input = mo.ui.text_area(placeholder="Paste HTML code...", full_width=True)
    html_input 
    return (html_input,)


@app.cell
def _(html_input):
    soup = BeautifulSoup(html_input.value, 'html5lib')
    cleaned_html = soup.get_text(separator=" ", strip=True)
    print(cleaned_html)
    return


@app.cell
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
