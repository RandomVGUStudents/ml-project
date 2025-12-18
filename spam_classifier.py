import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data acquisition & preprocessing
    """)
    return


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
    import tarfile
    import wget

    from bs4 import BeautifulSoup
    import mailparser
    import pandas as pd
    return BeautifulSoup, glob, hashlib, mailparser, os, pd, tarfile, wget


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


    def download_dataset(dataset_path, dataset_url):
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
def _(dataset_dir, dataset_source, download_dataset, hashlib, os):
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
                        print(f"File {_filename} is corrupted, redownloading dataset {_dataset_name}...")

                        download_dataset(_dataset_name, _dataset_info['url'])
                        os.remove(os.path.join(_dataset_path, _filename))

                        break

        else:
            print(f"Dataset {_dataset_name} not found, downloading...")
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


@app.cell
def _(idx, mailparser, mails):
    print(mails[idx.value])

    mail = mailparser.parse_from_file(mails[idx.value])
    print(mail.subject)
    print("---")
    print(mail.text_plain)
    print("---")
    print(mail.text_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load whole dataset
    """)
    return


@app.cell
def _(dataset_dir, dataset_source, mailparser, os, pd):
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


    df = pd.DataFrame(data)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocess email
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pre-cleanup
    """)
    return


@app.cell
def _():
    def artifact_cleanup(text):
        artifact = "--DeathToSpamDeathToSpamDeathToSpam--"
        return text.replace(artifact, "")

    def footer_cleanup(text):
        pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### HTML cleanup
    """)
    return


@app.cell
def _(BeautifulSoup, html_string, pd):
    def html_cleanup(html):
        if pd.isna(html_string):
            return ""
        soup = BeautifulSoup(html, 'html5lib')
        return soup.get_text(separator=" ", strip=True)
    return (html_cleanup,)


@app.cell
def _(df, html_cleanup):
    df['html'].apply(html_cleanup)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Text cleanup
    """)
    return


@app.function
def text_cleanup(text):
    pass


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
    - Number of links
    - Number of special characters
    - Captials ratio
    - Email size
    """)
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
    html_input = mo.ui.text_area(placeholder="Paste HTML code...")
    html_input 
    return (html_input,)


@app.cell(hide_code=True)
def _(BeautifulSoup, html_input):
    soup = BeautifulSoup(html_input.value, 'html5lib')
    cleaned_html = soup.get_text(separator=" ", strip=True)
    print(cleaned_html)
    return


if __name__ == "__main__":
    app.run()
