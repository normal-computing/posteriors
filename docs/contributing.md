If you want to add a new algorithm, example, feature, report or fix a bug, please open 
an [issue on GitHub](https://github.com/normal-computing/uqlib/issues). 
We'd love to have you involved in any capacity!

If you are interested in contributing to `uqlib`, please follow these steps:

1. [Fork the repo from GitHub](https://github.com/normal-computing/uqlib/fork)
and clone it locally:
```
git clone git@github.com/YourUserName/uqlib.git
cd uqlib
```
2. Install the development dependencies and pre-commit hooks:
```
pip install -e '.[test, docs]'
pre-commit install
```
3. **Add your code. Add your tests. Update the docs if needed. Party on!**
4. Make sure to run the tests and linter:
```
python -m pytest
pre-commit run --all-files
```
5. Commit your changes and push your new branch to your fork.
6. Open a [pull request on GitHub](https://github.com/normal-computing/uqlib/pulls).

!!! note
    Feel free to open a draft PR to discuss changes or get feedback.