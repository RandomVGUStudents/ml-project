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

    from dateutil import parser
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
        typing,
        wget,
    )


@app.cell
def _(spacy):
    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("merge_entities", after="ner")

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "TIME", "pattern": [{"SHAPE": "dd:dd:dd"}]},
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
            'checksum': "6ab4e23a696aeac72ad3b38396666a25"
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
def _(
    dataset_dir,
    dataset_source,
    load_dataset,
    mailparser,
    mo,
    os,
    pd,
    save_dataset,
):
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
    return (save_dataset,)


@app.cell
def _(hashlib):
    hashlib.md5(open("./datasets/orig.gzip", 'rb').read()).hexdigest()
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


@app.cell
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
    - Mailling list (or some name) in subject that is not filtered (like ilug)
    - Entity labels exclusion: "CARDINAL", "ORDINAL"
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
    artifact_cleanup,
    df,
    email_pattern,
    html_cleanup,
    idx,
    merge_text_html,
    mo,
    nlp,
    random_cleanup,
    spacy,
    subject_cleanup,
):
    _tfidf = TfidfVectorizer()
    _entry = df.iloc[idx.value]

    _clean_subj = subject_cleanup(_entry['subject'])
    _clean_html = html_cleanup(_entry['html'])
    _merged_text = merge_text_html(_entry['text'],_entry['html'])
    _text = f"{_clean_subj}\n{_merged_text['str']}"

    _text_2 = random_cleanup(_text)
    _text_3 = artifact_cleanup(_text_2)


    _doc = nlp(_text_3)


    with _doc.retokenize() as retokenizer:
        for ent in _doc.ents:
            if ent.label_ in ["DATE", "TIME", "MONEY"]:
                retokenizer.merge(ent)

        for match in email_pattern.finditer(_doc.text):
            span = _doc.char_span(match.start(), match.end())
            if span is not None:
                retokenizer.merge(span)

        # for match in url_pattern.finditer(_doc.text):
        #     span = _doc.char_span(match.start(), match.end())
        #     print(span)
        #     if span is not None:
        #         retokenizer.merge(span)


    _table2 = mo.ui.table(
        data=[{
            'Token': token.text,
            'Lemma': token.lemma_,
            'PoS': token.pos_,
            'Tag': token.tag_,
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


    def token_processor(token: spacy.tokens.Token) -> str:
        if token.is_stop:
            return ""

        if token.like_url:
            return "_URL_"
        if (token.like_email) or (token.text.startswith("<") and "@" in token.text and token.text.endswith(">")):
            return "_EMAIL_"

        if token.ent_type_ not in ["", "CARDINAL", "ORDINAL"]:
            return f"_{token.ent_type_}_"

        if token.pos_ == "PROPN":
            return "_PROPN_"
        if token.pos_ == "NUM":
            return "_NUM_"
        else:
            return token.lemma_.lower()


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
                    mo.md(f"{_clean_subj}\n{_merged_text['str']}".replace("\n", " ")),
                    mo.md(_text_3.replace("\n", " ")),
                    mo.md(_newtext.replace("\n", " "))
                ],
                widths="equal"
            ),
            mo.md(f"URL count: {_merged_text['link_count']}"), # If there's still URLs, add it to _URL_ token count.
            mo.md(f"Stop word ratio: {len([tok for tok in _doc if tok.is_stop]) / _doc.__len__()}"),
            mo.hstack([
                _table,
                _table2,
                _table3
            ])
        ])
    return


@app.cell
def _(Counter, sqrt):
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
    return (cosine_similarity,)


@app.cell
def _(BeautifulSoup, cosine_similarity, url_pattern):
    def merge_text_html(text: str, html: str, threshold = 0.95) -> object:
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
                print(similarity)
            
                if similarity >= threshold:
                    merged_text['str'] = text_no_url
            
                else:
                    merged_text['str'] = f"{text_no_url}\n{clean_html}"
                    merged_text['link_count'] += len(soup.find_all('a'))

        else:
            merged_text['str'] = url_pattern.sub(string=text, repl="")

        return merged_text
    return (merge_text_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pre-cleanup
    """)
    return


@app.cell
def _(BeautifulSoup, pd, re, re2):
    unamed_pattern = re2.compile(
        r"[-#*_=+]{3,}"
    )

    url_pattern = re.compile(
        r"((http|ftp|https):\/\/)?([\w_-]+(?:\.[\w_-]+)*\.[a-zA-Z_-][\w_-]+)([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
    )

    email_pattern = re.compile(
        r"<[^>]+@[^>]+>"
    )

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


    def footer_cleanup(text):
        pass


    def trim(text):
        return text.strip("\n")


    def html_cleanup(html):
        if pd.isna(html):
            return ""

        soup = BeautifulSoup(html, 'html5lib')

        return soup.get_text(separator=" ", strip=True)
    return (
        artifact_cleanup,
        email_pattern,
        html_cleanup,
        random_cleanup,
        subject_cleanup,
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
    - Capitals ratio
    - Email size (maybe not)
    - Reply ratio, reply depth count
    - Stop word ratio
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Number of links
    """)
    return


@app.cell
def _(BeautifulSoup, typing, url_pattern):
    def count_links(text: typing.Optional[str] = None, html: typing.Optional[str] = None):
        if text is not None:
            return len(url_pattern.findall(text))

        if html is not None:
            soup = BeautifulSoup(html, 'html5lib')
            soup.find_all('a')
            return 
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
