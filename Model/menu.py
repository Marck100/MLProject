# Element of the menu (you can provide title, description and action handler)
class MenuItem:
    title: str
    description: str

    def __init__(self, title, description, action):
        self.title, self.description = title, description
        self.action = action

    def action(self):
        pass

    def __call__(self):
        self.action()

    def __str__(self):
        title, description = self.title, self.description
        string = f'{title}\n   {description}'
        return string


class Menu:
    _items: [MenuItem]

    def __init__(self, items: [MenuItem] = None):
        self._items = list() if items is None else items

    def addItem(self, item: MenuItem):
        items = self._items
        item.code = len(items) + 1
        items.append(item)

    def show(self):
        print('------MENU------')
        for index, item in enumerate(self._items):
            print(f'{index + 1}. {item}')
        print('Type -1 to exit')
        print('----------------')

    def askForChoice(self):
        items = self._items

        print('Enter value:')
        choice = 0

        while choice != -1 and (choice < 1 or choice > len(items)):
            choice = int(input('? '))

        if choice == -1:
            return -1

        choice -= 1
        items[choice]()

        return choice


if __name__ == '__main__':
    item1 = MenuItem('Choice 1', 'Just an example', lambda: print('Choice 1'))
    item2 = MenuItem('Choice 2', 'Just an example', lambda: print('Choice 2'))

    menu = Menu()
    menu.addItem(item1)
    menu.addItem(item2)
    menu.show()
    menu.askForChoice()
