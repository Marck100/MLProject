from Model.menu import MenuItem, Menu
from Model.analyzer import Analyzer

analyzer = Analyzer('Resources/Customers.csv')
analyzer.load()

flag = 1

stats_menu_item = MenuItem(
    'Stats',
    'Show stats (number of records and columns, duplicates, redundant elements)',
    lambda: analyzer.showStats()
)

menu = Menu([stats_menu_item])

while 1:
    menu.show()
    if menu.askForChoice() == -1:
        break
    print()
