import opendatasets as od


class kaggle_api():

    def __init__(self, link):
        self.link = link

    def download(self):
        return od.download(self.link)
