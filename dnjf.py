from collections import Counter
l = ['someone', 'let', 'me', 'out', 'of', 'the', 'cage','time','for','me','is','nothing','but','a','cage']
counter = Counter(l)

if __name__ == "__main__":
    print(counter)
    print(counter['me']) # 2
    print(counter['cage']) # 2
    print(dict(counter))