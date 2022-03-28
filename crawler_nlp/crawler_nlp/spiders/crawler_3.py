import scrapy

class EnglishSpider(scrapy.Spider):
    name = "WikiSpider"
    start_urls = [
        "http://en.volupedia.org/wiki/Chagas_disease",
    ]
    def parse(self, response):
        for content in  response.xpath('//div[@class="mw-parser-output"]/p/text() | //div['
                                       '@class="mw-parser-output"]/p/a/text() | //div['
                                       '@class="mw-parser-output"]/p/b/text() | //div['
                                       '@class="mw-parser-output"]/p/i/a/text()').getall():
                yield {
                    'p': content,
                }
        for page in response.xpath('//div[@class="mw-parser-output"]/p/a/@href | //div['
                                   '@class="mw-parser-output"]/p/i/a/@href').getall():
            page = response.urljoin('http://en.volupedia.org'+page)
            yield scrapy.Request(page, callback=self.parse)




