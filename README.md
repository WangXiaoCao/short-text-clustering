# short-text-clustering
> The project is done using Jupyter Notebook with Python 3.7, PyTorch 1.1.0, sklearn, ...

Text clustering using `sklearn` tfidf and homemade k-means algorithm implemented on `PyTorch`.

## Directory Structure

```
project
├─data
│  ├─test_tokens.json               Test dataset
│  ├─train_tokens.json              Training dataset
│  ├─train_topics.json              Training evaluation labels
│  └─vocab.json                     Vocabulary
├─src
│  ├─main.py                        cli tool  
│  └─train_clustering.ipynb         Main notebook  
│
...
```

## Jupyter Notebook

See the [notebook](./src/train_clustering.ipynb) to see details of the algorithms and implementation.

## License

MIT, see the [LICENSE](/LICENSE) file for details.