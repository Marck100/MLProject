# Element of the menu (you can provide title, description and action handler)
class MenuItem:
    title: str
    description: str

    def __init__(self, title, description, action):
        self.title, self.description = title, description
        self.action = action

    # Action executed on option selection
    def action(self):
        pass

    def __call__(self):
        self.action()

    # Readable print
    def __str__(self):
        title, description = self.title, self.description
        string = f'{title}\n   {description}'
        return string


# Display menu with options
class Menu:
    _items: [MenuItem]

    def __init__(self, items: [MenuItem] = None):
        self._items = list() if items is None else items

    # Print menu with options
    def show(self):
        print('------MENU------')
        for index, item in enumerate(self._items):
            print(f'{index + 1}. {item}')
        print('Type -1 to exit')
        print('----------------')

    # Get input and execute menu item choice
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

    menu = Menu([item1, item2])
    menu.show()
    menu.askForChoice()
