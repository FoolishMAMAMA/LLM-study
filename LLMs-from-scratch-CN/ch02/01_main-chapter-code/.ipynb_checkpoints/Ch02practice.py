import os##导入os库    
import urllib.request ##导入request库

if not os.path.exists("the-verdict.txt"):##如果文件不存在则创建，防止因文件已存在而报错
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)##从指定的地点读取文

    with open("the-verdict.txt", "r", encoding="utf-8") as f:##以只读方式打开文件
        raw_text=f.read()
    print("Total number of characters in the dataset:", len(raw_text))##打印数据集中的总字符数  
    print(raw_text[:99])##打印前99个字符

    