# NaturalLanguageProcessing
## crawler_nlp
python environment: 3.8

launch:
```
cd crawler_nlp/crawler_nlp
pip install scrapy
pip install scipy
scrapy crawl ChinaDailySpider -o contents.jl
scrapy crawl NovelSpider -o contents.jl
scrapy crawl WikiSpider -o contents.jl

```

## FNN_two_methods

* contents_test.jl is smaller than contents_small.jl

* In every condition, there is only 1 weights file reserved. 
* test and train
  * python file is for training
  * ipynb file is for testing

### Skip-gram

### 5-gram

## Transformer_NMT
* python transformer_nmt.py

## BLEU
collect data by hand
use the jupyter notebook to calculate

## EmotionKeyWordsExtract
use crawl.ipynb to crawl data and do the TF-IDF calculation
