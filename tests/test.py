with (open('tests/test.txt', 'r')) as f: 
    words = f.read().split()
    print(len(words))