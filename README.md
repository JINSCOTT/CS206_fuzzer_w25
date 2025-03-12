# Atomic fuzzer

* Install dependency
* pip install -r requirements.txt

## How to run
### To generate seeds
* Export Openai key
```bash
export OPENAI_API_KEY='yourkey'
```

```bash
python generate.py
```

### To run tests
```bash
pytest
```
### To run coverage
```bash
pytest --cov=torch --cov-report html
```
## Equivalent Operators in PyTorch Optimization Testing
* This is an investigation on speed of combination of operators befor and after compiling.
### PyTorch Optimization Testing Colab Link:

* https://colab.research.google.com/drive/1CgJsu5hUQKztL3k_NwYKiyO-Rrf3RbzN?authuser=1#scrollTo=WEdWJaewbgUJ
