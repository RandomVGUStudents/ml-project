# Machine Learning project

## IMPORTANT: Windows has a skill issue which will make it fail to parse some spam emails

Use Linux

<details>

### What's wrong?

Some emails has malformed timestamp (e.g. the year is set to 0102 to purposely mess with the system, like preventing itself to be parsed in this case).

But Linux doesn't care about this issue, it seems.

### I don't have Linux

In that case, some spam emails may not be parsed, reducing the amount of training data for the `spam` label.

</details>

## Me towards vibe coding

- No vibe coding
- No vibe coding
- No vibe coding

## Requirements

- `python` (duh).
Download [here.](https://www.python.org/downloads/) (You don't have that yet, really?)

<details>

### Just kidding

You didn't need to download that one. Continue.

</details>

- `uv` (makes life easier, I think). Run this command as Administrator (press `Windows + X`, then `A`, and paste this):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

<details>

<summary>No virus, I swear</summary>

### Don't believe that?

You have the right to, and indeed, it's good to be skeptical. Check out [`uv`'s download page](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

</details>

- `vscode` and `git`.
Please, I don't send code as zip files via Messenger, nor do I use Notepad.

<details>

<summary>Still don't want to?</summary>

### Okay...

What can I say.

</details>

## Installation

### With `git`:

```bash
git clone https://github.com/RandomVGUStudents/ml-project
```

### Without `git`:

1. Click the green button `< > Code` on the screen, and click `Download ZIP`.
2. Consider installing `git`, maybe?
3. 6
4. Extract

## Edit

In a VSCode window, or Powershell window, run:

```bash
uv run marimo edit .\spam_classifier.py
```

## TODO

- [x] Download dataset
- [ ] Preprocess & cleanup
- [ ] Tokenize & extract features
- [ ] Train
- [ ] Write report, presentation, etc.
