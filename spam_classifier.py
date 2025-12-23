import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import some libs
    """)
    return


@app.cell(hide_code=True)
def _():
    import glob
    import hashlib
    import os
    import re
    import re2
    import spacy
    import spacy_transformers
    import tarfile
    import typing
    import wget
    import mailparser
    import pandas as pd

    from bs4 import BeautifulSoup
    from collections import Counter
    from math import sqrt
    from sklearn.feature_extraction.text import TfidfVectorizer
    return (
        BeautifulSoup,
        Counter,
        TfidfVectorizer,
        glob,
        hashlib,
        mailparser,
        os,
        pd,
        re,
        re2,
        spacy,
        sqrt,
        tarfile,
        wget,
    )


@app.cell
def _(spacy):
    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("merge_entities", after="ner")

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "URL_TOK", "pattern": "_URL_"},
        {"label": "EMAIL_TOK", "pattern": "_EMAIL_ADDR_"}
    #    {"label": "URL_TOK", "pattern": "_url_"}
    #    {"label": "URL_TOK", "pattern": "_url_"}
    ]
    ruler.add_patterns(patterns)
    return (nlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data acquisition & preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define dataset source
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
            'count': 1397,
            'is_spam': True
        }
    }
    return (dataset_source,)


@app.cell
def _(os, tarfile, wget):
    dataset_dir = "./datasets/"


    def download_dataset(dataset_path: str, dataset_url: str):
        tmp = wget.download(dataset_url)

        with tarfile.open(tmp, "r:bz2") as tar:
            tar.extractall(dataset_dir)

        os.remove(os.path.join(dataset_path, "cmds"))
        os.remove(tmp)
    return dataset_dir, download_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check dataset integrity
    """)
    return


@app.cell
def _(dataset_dir, dataset_source, download_dataset, hashlib, mo, os):
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
def _(mo):
    mo.md(r"""
    ## Viewing some emails
    """)
    return


@app.cell(hide_code=True)
def _(glob, mo):
    mails = glob.glob("./datasets/*/*", recursive=True)
    idx = mo.ui.number(start=0, stop=len(mails) - 1, label="Number")
    return idx, mails


@app.cell(hide_code=True)
def _(idx):
    idx
    return


@app.cell(hide_code=True)
def _(idx, mailparser, mails, mo):
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
def _(mo):
    mo.md(r"""
    ## Load whole dataset
    """)
    return


@app.cell
def _():
    dataset_checkpoints = {
        'orig': {
            'description': "Dataset without preprocessing (6046 entries)",
            'checksum': "5cb36b7b38b8dc643b1dbc099c7ea598"
        }
    }
    return (dataset_checkpoints,)


@app.cell
def _(dataset_checkpoints, dataset_dir, hashlib, os, pd):
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


@app.cell
def _(dataset_dir, dataset_source, load_dataset, mailparser, mo, os, pd):
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

        mo.md(e)
        mo.md("Loading from files...")

        df = pd.DataFrame(data)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save dataset
    """)
    return


@app.cell
def _(dataset_dir, hashlib, mo, os, pd):
    def save_dataset(df: pd.DataFrame, checkpoint_name: str):
        path = os.path.join(dataset_dir, f"{checkpoint_name}.gzip")
        df.to_parquet(
            path=path,
            compression='gzip'
        )

        hash = hashlib.md5(open(path, 'rb').read()).hexdigest()
        mo.md(f"File: {path}\nMD5: {hash}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocess email

    1. Remove duplicates from raw dataset
    2. Find and replace certain tokens with regex (email, addresses, phone number, etc.)
    3. Find and replace proper noun tokens
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove duplicates
    """)
    return


@app.cell
def _(df):
    df.drop_duplicates(subset=['text', 'html'], inplace=True)
    return


@app.cell
def _(idx):
    idx
    return


@app.cell(hide_code=True)
def _(
    TfidfVectorizer,
    cosine_similarity,
    df,
    html_cleanup,
    idx,
    mo,
    nlp,
    random_cleanup,
    re,
    replace_email_token,
    replace_url_token,
    spacy,
    url_pattern,
):
    _tfidf = TfidfVectorizer()
    _entry = df.iloc[idx.value]
    _text = f"{_entry['subject']}\n{_entry['text']}"
    _clean_html = html_cleanup(_entry['html'])

    # Remove sponsor and mailing list
    # _sponsor_pattern = re2.compile(
    #     r"(?m)^[> \t]*-{5,}\s+This sf\.net email is sponsored by:.*(?s:.*?)(?=(?m)^[> \t]*_{5,}|\Z)"
    # )
    # 
    # _text = re2.sub(
    #     pattern=_sponsor_pattern,
    #     text=_text,
    #     repl="_______________________________________________"
    # )
    # 
    # _maillist_pattern = re2.compile(
    #     r"(?m)^[> \t]*_{5,}\s*[\r\n]+[> \t]*.* mailing list(?:[\r\n]+[> \t]*.*){0,4}"
    # )
    # 
    # _text = re2.sub(
    #     pattern=_maillist_pattern,
    #     text=_text,
    #     repl=""
    # )

    # Replace email address tokens
    _text_1 = replace_email_token(_text)

    # Replace URL tokens
    _text_2 = replace_url_token(_text_1)

    # What
    _text_3 = random_cleanup(_text_2)

    _doc = nlp(_text_3)


    _table2 = mo.ui.table(
        data=[{
            'Token': token.text,
            'Lemma': token.lemma_,
            'PoS': token.pos_,
            'Tag': token.tag_,
            'Dependency': token.dep_,
            'Stop word': token.is_stop
        } for token in _doc],
        pagination=True
    )


    def token_processor(token: spacy.tokens.Token) -> str:
        if token.is_stop:
            return ""
        if token.text in ["_URL_", "_EMAIL_ADDR_"]:
            return token.text
        if token.pos_ == "PROPN":
            return "_PROPN_"
        if token.pos_ == "NUM":
            return "_NUM_"
        else:
            return token.lemma_


    _newtext = " ".join(
        [token_processor(token) for token in _doc]
    )


    _results = _tfidf.fit_transform([_newtext])
    _vocab = [{'word': _word, 'idx': _idx} for _word, _idx in _tfidf.vocabulary_.items()]
    _table = mo.ui.table(data=_vocab, pagination=True)


    mo.vstack(
        [
            mo.md(f"Label: {"Ham" if _entry['label'] == 0 else "Spam"}"),
            mo.hstack(
                [
                    mo.md(f"""```text
    {_text}
    ```
    """),
                    mo.md(f"""```text
    {_text_1}
    ```
    """),
                    mo.md(f"""```text
    {_text_2}
    ```
    """),
                    mo.md(f"""```text
    {_text_3}
    ```
    """),
                    mo.md(f"""```text
    {_newtext}
    ```
    """),
                    mo.md(f"""```html
    {_entry['html']}
    ```
    """),
                    mo.md(f"""```text
    {_clean_html}
    ```
    """)
                ]),
            mo.md(f"Cosine similarity: {cosine_similarity(_entry['text'], _clean_html)}"),
            mo.md(f"URL count: {len(re.findall(url_pattern, f"{_entry['subject']}\n{_entry['text']}"))}"),
            mo.hstack([
                _table2,
                _table
            ])
        ])
    return


@app.cell
def _(Counter, sqrt):
    def cosine_similarity(s1: str, s2: str) -> float:
        # Convert strings to character frequency vectors
        vec1 = Counter(s1)
        vec2 = Counter(s2)
    
        # Calculating cosine similarity
        dot_product = sum(vec1[ch] * vec2[ch] for ch in vec1)
        magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
        magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))
        return dot_product / (magnitude1 * magnitude2)
    return (cosine_similarity,)


@app.cell
def _(cosine_similarity, html_cleanup):
    def merge_text_html(text: str, html: str, threshold = 0.95) -> str:
        if (len(text) == 0 and len(html) != 0):
            return html_cleanup(html)
        elif len(html) == 0:
            return text

        clean_html = html_cleanup(html)

        if cosine_similarity(text, html) > threshold:
            return text
        else:
            return f"{text}\n{clean_html}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pre-cleanup
    """)
    return


@app.cell
def _(BeautifulSoup, pd, re, re2):
    # url_pattern = re2.compile(
    #     r"(https?:\/\/)?[a-zA-Z0-9][a-zA-Z0-9.\-]*\.[a-zA-Z]{2,}(:\d+)?(\/[a-zA-Z0-9.\-_\~:/?#[\]@!$&'()*+,;=%]*)?"
    # )

    url_pattern = re.compile(
        "((http|ftp|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    )
    
    email_pattern = re2.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )

    unamed_pattern = re2.compile(
        r"[-#*_=+]{3,}"
    )

    
    def artifact_cleanup(text):
        artifact = "--DeathToSpamDeathToSpamDeathToSpam--"
        return text.replace(artifact, "")


    def replace_url_token(text: str) -> str:
        return re.sub(
            pattern=url_pattern,
            string=text,
            repl="_URL_"
        )


    def replace_email_token(text: str) -> str:
        return re2.sub(
            pattern=email_pattern,
            text=text,
            repl="_EMAIL_ADDR_"
        )


    def random_cleanup(text: str) -> str:
        return re2.sub(
            pattern=unamed_pattern,
            text=text,
            repl=""
        )


    def footer_cleanup(text):
        pass


    def trim(text):
        return text.strip("\n")


    def html_cleanup(html):
        if pd.isna(html):
            return ""

        soup = BeautifulSoup(html, 'html5lib')

        # Add tokens
        append_tok = " ".join(["_URL_"] * len(soup.find_all('a')))
        print(append_tok)
    
        return f"{soup.get_text(separator=" ", strip=True)}\n{append_tok}"
    return (
        html_cleanup,
        random_cleanup,
        replace_email_token,
        replace_url_token,
        url_pattern,
    )


@app.cell
def _(df, html_cleanup):
    df['html'].apply(html_cleanup)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Features engineering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Additional features to consider

    - Contains HTML (boolean)
    - Number of links (share the same token with links in text)
    - ~~Number of special characters~~ Might not be a good feature
    - Captials ratio
    - Email size (maybe not)
    - Reply ratio, reply depth count
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Number of links
    """)
    return


@app.cell
def _(BeautifulSoup):
    def count_links(html):
        soup = BeautifulSoup(html, 'html5lib')
        return len(soup.find_all('a'))
    return (count_links,)


@app.cell
def _(count_links, df):
    df['html'].apply(count_links)
    return


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
    mo.md(r"""
    # Random stuff
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(df, mo):
    mo.ui.dataframe(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Beautiful Soup playground
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    html_input = mo.ui.text_area(placeholder="Paste HTML code...", full_width=True)
    html_input 
    return (html_input,)


@app.cell(hide_code=True)
def _(BeautifulSoup, html_input):
    soup = BeautifulSoup(html_input.value, 'html5lib')
    cleaned_html = soup.get_text(separator=" ", strip=True)
    print(cleaned_html)
    return


@app.cell
def _(count_links, html_input):
    print(count_links(html_input.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RE2 playground
    """)
    return


@app.cell
def _(mo):
    text_input = mo.ui.text_area(placeholder="Paste text...", full_width=True)
    text_input
    return (text_input,)


@app.cell
def _(mo):
    regex_input = mo.ui.text(placeholder="Regex", full_width=True)
    regex_input
    return (regex_input,)


@app.cell
def _(re2, regex_input, text_input):
    re2.sub(pattern=regex_input.value, text=text_input.value, repl="")
    return


if __name__ == "__main__":
    app.run()
