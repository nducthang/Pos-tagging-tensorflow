from pyvi import ViTokenizer, ViPosTagger

if __name__ == '__main__':
    print(ViTokenizer.tokenize(u"Chiếm đến lượng hàng hoá lưu thông trên đường biển của Thế giới"))

    print(ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội")))
