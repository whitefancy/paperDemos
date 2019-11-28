import os

data_dir='../data/semeval2014/'

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(dir_path+"test_QA_M.csv","w",encoding="utf-8") as g:
    with open(data_dir+"Restaurants_Test_Gold.xml","r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            category=[]
            polarity=[]
            if "<sentence id" in s:
                left=s.find("id")
                right=s.find(">")
                id=s[left+4:right-1]
                while not "</sentence>" in s:
                    if "<text>" in s:
                        left=s.find("<text>")
                        right=s.find("</text>")
                        text=s[left+6:right]
                    if "aspectCategory" in s:
                        left=s.find("category=")
                        right=s.find("polarity=")
                        category.append(s[left+10:right-2])
                        left=s.find("polarity=")
                        right=s.find("/>")
                        polarity.append(s[left+10:right-2])
                    s=f.readline().strip()
                if "price" in category:
                    g.write(id+"\t"+polarity[category.index("price")]+"\t"+"what do you think of the price reason worth cheap expens wine qualiti better overpric pricey drink lunch valu of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the price reason worth cheap expens wine qualiti better overpric pricey drink lunch valu of it ?" + "\t" + text + "\n")
                if "anecdotes/miscellaneous" in category:
                    g.write(id+"\t"+polarity[category.index("anecdotes/miscellaneous")]+"\t"+"what do you think of the place time go recommend friend night live dinner come wait review experi of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the place time go recommend friend night live dinner come wait review experi of it ?" + "\t" + text + "\n")
                if "food" in category:
                    g.write(id+"\t"+polarity[category.index("food")]+"\t"+"what do you think of the food delici dish menu pizza wine fresh order tast chicken sushi of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the food delici dish menu pizza wine fresh order tast chicken sushi of it ?" + "\t" + text + "\n")
                if "ambience" in category:
                    g.write(id+"\t"+polarity[category.index("ambience")]+"\t"+"what do you think of the place atmospher decor ambienc romant tabl littl ambianc seat small look of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the place atmospher decor ambienc romant tabl littl ambianc seat small look of it ?" + "\t" + text + "\n")
                if "service" in category:
                    g.write(id+"\t"+polarity[category.index("service")]+"\t"+"what do you think of the servic staff friendli tabl wait time attent waiter come order seat reserv of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the servic staff friendli tabl wait time attent waiter come order seat reserv of it ?" + "\t" + text + "\n")
            else:
                s = f.readline().strip()


with open(dir_path+"train_QA_M.csv","w",encoding="utf-8") as g:
    with open(data_dir+"Restaurants_Train.xml","r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            category=[]
            polarity=[]
            if "<sentence id" in s:
                left=s.find("id")
                right=s.find(">")
                id=s[left+4:right-1]
                while not "</sentence>" in s:
                    if "<text>" in s:
                        left=s.find("<text>")
                        right=s.find("</text>")
                        text=s[left+6:right]
                    if "aspectCategory" in s:
                        left=s.find("category=")
                        right=s.find("polarity=")
                        category.append(s[left+10:right-2])
                        left=s.find("polarity=")
                        right=s.find("/>")
                        polarity.append(s[left+10:right-1])
                    s=f.readline().strip()
                if "price" in category:
                    g.write(id+"\t"+polarity[category.index("price")]+"\t"+"what do you think of the price reason worth cheap expens wine qualiti better overpric pricey drink lunch valu of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the price reason worth cheap expens wine qualiti better overpric pricey drink lunch valu of it ?" + "\t" + text + "\n")
                if "anecdotes/miscellaneous" in category:
                    g.write(id+"\t"+polarity[category.index("anecdotes/miscellaneous")]+"\t"+"what do you think of the place time go recommend friend night live dinner come wait review experi of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the place time go recommend friend night live dinner come wait review experi of it ?" + "\t" + text + "\n")
                if "food" in category:
                    g.write(id+"\t"+polarity[category.index("food")]+"\t"+"what do you think of the food delici dish menu pizza wine fresh order tast chicken sushi of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the food delici dish menu pizza wine fresh order tast chicken sushi of it ?" + "\t" + text + "\n")
                if "ambience" in category:
                    g.write(id+"\t"+polarity[category.index("ambience")]+"\t"+"what do you think of the place atmospher decor ambienc romant tabl littl ambianc seat small look of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the place atmospher decor ambienc romant tabl littl ambianc seat small look of it ?" + "\t" + text + "\n")
                if "service" in category:
                    g.write(id+"\t"+polarity[category.index("service")]+"\t"+"what do you think of the servic staff friendli tabl wait time attent waiter come order seat reserv of it ?"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "what do you think of the servic staff friendli tabl wait time attent waiter come order seat reserv of it ?" + "\t" + text + "\n")
            else:
                s = f.readline().strip()