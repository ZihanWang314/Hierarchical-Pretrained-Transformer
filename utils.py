import os

class Logger:
    def __init__(self, path, sep=' ', end='\n'):
        self.path = os.path.join('log', path)
        self.sep = sep
        self.end = end
        if not os.path.exists('log'):
            os.mkdir('log')
        
    def log(self, *args):
        output_list = []
        for arg in args:
            output_list.append(str(arg))
        output_list.append(self.end)
        output = self.sep.join(output_list)
        with open(self.path, 'a', encoding = 'utf-8') as file:
            file.write(output)