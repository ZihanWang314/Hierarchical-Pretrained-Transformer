
levels = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'p': 5, 'li': 6, 'tr': 6}

def get_level(text):
    return levels[text[1:3].strip('>')]

class TxtNode:
    def __init__(self, text):
        self.text = text
        self.level = get_level(text)
        self.children = []
        self.parent = None

    def get_nodes_list(self):
        if self.children == []:
            return [self]
        else:
            nodes = [self]
            for child in self.children:
                nodes += child.get_nodes_list()
            return nodes


    def __str__(self, indent = 0):
        string = self.text[:10]
        string = ' ' * indent + string
        if self.children == []:
            return string
        else:
            return '\n'.join([string] + [node.__str__(indent + 4) for node in self.children])

    __repr__ = __str__


    def __len__(self):
        nodes = self.get_nodes_list()
        return sum(len(i.text) for i in nodes)

        
    def copy(self): # deep copy 
        newnode = TxtNode(self.text, self.tokenizer)
        for child in self.children:
            newnode.children.append(child.copy())
        for child in newnode.children:
            child.parent = newnode
        return newnode
        
def qa_from_document(doc_object):
    title, url, content = doc_object.values()
    content = 