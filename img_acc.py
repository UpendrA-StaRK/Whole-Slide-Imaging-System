#IMAGE ACCUMULATOR

class d_que:
    def _init_(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0
    
    def append(self, item):
        self.items.append(item)

    def popleft(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            return None
                
    def get_item(self,index):
        if index < len(self.items):
            return self.items[index]
        else:
            return None       
    def remove_rear(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            return None

    def size(self):
        return len(self.items)
