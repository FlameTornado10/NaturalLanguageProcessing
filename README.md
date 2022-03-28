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
