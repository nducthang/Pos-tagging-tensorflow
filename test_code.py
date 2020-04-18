from pyvi import ViTokenizer, ViPosTagger

print(ViTokenizer.tokenize(u"Chiếm đến lượng hàng hoá lưu thông trên đường biển của Thế giới"))

print(ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội")))
