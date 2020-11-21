import re
def createdata(p1,p2,p3):
	f1 = open(p1,"r")
	f2 = open(p2,"w")
	f3 = open(p3,"w")
	for line in f1:
		s = line.split("####")
		words = s[0].split()
		sent = s[0]+'\n'
		f2.write(sent)
		tups = re.findall(r'\((.*?)\)',s[1])
		aop = ''
	#	print(line)
		for t in tups:
			t1 = t.split("],")
			aspects =  " ".join([words[int(k)] for k in t1[0][1:].split(',')])
			opinions = " ".join([words[int(k)] for k in t1[1][2:].split(',')])
			polar = t1[2][-4:-1]
			aop += (' '+aspects + ' ; ')
			aop += (opinions + ' ; ')
			aop += (polar + ' |')
		aop = aop[1:-2]+'\n'
		f3.write(aop)
	f1.close()
	f2.close()
	f3.close()
	return
def main():
    createdata("train_triplets.txt","train.sent","train.tup")
    createdata("test_triplets.txt","test.sent","test.tup")
    createdata("dev_triplets.txt","dev.sent","dev.tup")


if __name__=="__main__":
    main()
