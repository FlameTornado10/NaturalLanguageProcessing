import scrapy

class EnglishSpider(scrapy.Spider):
    name = "NovelSpider"
    start_urls = [
        # "https://allnovel.net/p-s-i-still-love-you-to-all-the-boys-i-ve-loved-before-2/page-1.html",
        # "https://allnovel.net/a-ruthless-proposition/page-1.html",
        # "https://allnovel.net/a-husband-s-regret-unwanted-2/page-1.html",
        # "https://allnovel.net/wired-buchanan-renard-13/page-1.html",
        "https://allnovel.net/to-all-the-boys-i-ve-loved-before-to-all-the-boys-i-ve-loved-before-1/page-1.html",
        "https://allnovel.net/whitney-my-love-westmoreland-saga-2/page-1.html",
        "https://allnovel.net/a-kingdom-of-dreams-westmoreland-saga-1/page-1.html",
        "https://allnovel.net/mine-till-midnight-the-hathaways-1/page-1.html",
    ]
    def parse(self, response):
        for content in response.xpath('//div[@class="content_novel"]//p/text()').getall():
            yield {
                'p': content,
            }
        next_page = response.xpath('//div[@id="detail-page"]//a[last()]/@href').get()
        if next_page is not None:
            next_page = response.urljoin('https://allnovel.net'+next_page)
            yield scrapy.Request(next_page, callback=self.parse)



