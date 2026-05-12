```
Total number of searches: 1

Search 1: pathlib Path.walk added Python version CPython PR
Finding: Python 3.12 introduced `pathlib.Path.walk()`; the corresponding CPython PR is #119573.

Final Answer:

First introduced in version: 3.12

Corresponding PR: 119573

Evidence Chain:

- "pathlib — Object-oriented filesystem paths — Python 3.12.13 documentation" + "Added in version 3.12." + https://docs.python.org/3.12/library/pathlib.html

- "PR #119573 GH-119169: Implement `pathlib.Path.walk()` using `os.walk()` - SemanticDiff" + "Merged" + https://app.semanticdiff.com/gh/python/cpython/pull/119573/overview

Reasoning:
The Python 3.12 documentation explicitly states that `Path.walk()` was added in version 3.12. The CPython GitHub pull request #119573, titled "Implement `pathlib.Path.walk()` using `os.walk()`," was merged into the main branch, indicating the integration of this feature.
```