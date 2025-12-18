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
    import email
    import glob
    import hashlib
    import os
    import tarfile
    import wget

    from email.parser import Parser

    import mailparser
    import pandas as pd
    return glob, hashlib, mailparser, os, tarfile, wget


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    ### Define dataset source
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
def _():
    ### Check dataset integrity
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
def _(dataset_dir, dataset_source, mailparser, os):
    data = {
        "subject": [],
        "text": [],
        "html": [],
        "label": []
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
                    data["subject"].append(_mail.subject)
                    data["text"].append(_mail.text_plain)
                    data["html"].append(_mail.text_html)
                    data["label"].append(int(_dataset_info['is_spam']))
    return


@app.function
def preprocess_email(text):
    pass


@app.cell(hide_code=True)
def _():
    ### Emails with text only
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocess emails
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random stuff imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
