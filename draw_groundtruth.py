
from PIL import Image, ImageDraw as D

i=Image.open("/home/ncl/ttmanh/September/fasterrcnn-pytorch-training-pipeline/data_test/0501130731_1_11314.jpg")
draw=D.Draw(i)
draw.rectangle([(1036.859375,427),(1273.859375,693)],outline="green")
draw.rectangle([(559.859375,410),(790.859375,706)],outline="green")
draw.rectangle([(1040.859375,91),(1272.8593750000002,380)],outline="green")


# save the image
i.save("/home/ncl/ttmanh/September/fasterrcnn-pytorch-training-pipeline/data_test/0501130731_1_11314_test_1.jpg")