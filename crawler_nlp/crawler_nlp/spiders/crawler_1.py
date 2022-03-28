import scrapy

class EnglishSpider(scrapy.Spider):
    name = "ChinaDailySpider"
    start_urls = [
        "http://www.chinadaily.com.cn/world",
        "http://www.chinadaily.com.cn/business",
        "http://www.chinadaily.com.cn/life",
        "http://www.chinadaily.com.cn/culture",
        "http://www.chinadaily.com.cn/travel",
        "http://www.chinadaily.com.cn/sports",
        "http://www.chinadaily.com.cn/opinion",
        "http://www.chinadaily.com.cn/regional",
        # "http://www.chinadaily.com.cn/a/202203/21/WS6237b6c5a310fd2b29e520f4.html",
    ]
    def parse(self, response):
        for content in  response.xpath('//div[@id="Content"]/p/text()').getall():
                yield {
                    'p': content,
                }
        for page in response.xpath('//div[@class="main"]//a[@shape="rect" and @target="_blank" and contains(@href, '
                                   '"html")]/@href').getall():
            page = response.urljoin(page)
            yield scrapy.Request(page, callback=self.parse)
        next_page = response.xpath('//div[@id="div_currpage"]/a[@class="pagestyle" and normalize-space(text('
                                   '))="Next"]/@href').get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)



