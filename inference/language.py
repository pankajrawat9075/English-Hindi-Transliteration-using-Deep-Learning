class Language:
    def __init__(self, name):
        self.name = name
        self.char2index = {'#': 0, '$': 1, '^': 2}   # '^': start of sequence, '$' : unknown char, '#' : padding
        self.index2char = {0: '#', 1: '$', 2: '^'}
        self.vocab_size = 3  # Count

    def addWord(self, word):
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.vocab_size
            self.index2char[self.vocab_size] = char
            self.vocab_size += 1

    def encode(self, s):
        return [self.char2index[ch] for ch in s]

    def decode(self, l):
        return ''.join([self.index2char[i] for i in l])

    def vocab(self):
        return self.char2index.keys()
