import xlwt
import os
import detect
path = "classesnames/" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
txts = []
j = 0
g = 0


for file in files: #遍历文件夹

    position = path + str(g) +".txt" #构造绝对路径，"\\"，其中一个'\'为转义符
    print (position)
    with open(position, "r",encoding='utf-8') as f:    #打开文件
        data = f.read()   #读取文件
        txts.append(data)
        print(data)
        g += 1


# 新建一个workbok
workbook = xlwt.Workbook(encoding='utf-8')


# 加入一个工作表
sheet1 = workbook.add_sheet('Sheet1')

# 向单元格写入数据
# sheet1.write(行号，列号，数据)


# 定义表头（第一行）
fields = ['id',  '类别']
idx = 0
for x in fields:
    sheet1.write(0, idx, x)
    idx = idx + 1

# 写入数据
for i in range(1, 5001):
    col = 0
    dt = ['%d' % (i-1), txts[j]]
    for x in fields:
        sheet1.write(i, col, dt[col])
        col = col + 1
    j += 1
# 保存Excel文件
workbook.save("分类best2.csv")

