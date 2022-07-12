# Implement HQ9+ interpreter on Python
class HQ9Plus:
    def __init__(self):
        self.cnt = 0

    def execute(self, code):
        # check syntax
        for c in code:
            if c not in ['H', 'Q', '9', '+']:
                print('Syntax error')
                return
        # execute code
        for c in code:
            if c == 'H':
                print('Hello, world!')
            elif c == 'Q':
                print(code)
            elif c == '9':
                for beer in range(99, 1, -1):
                    print(beer, 'bottles of beer on the wall,', beer, 'bottles of beer.')
                    print('Take one down and pass it around,', beer-1, 'bottles of beer on the wall.')
                    print()
                print('1 bottle of beer on the wall, 1 bottle of beer.')
                print('Take one down and pass it around, no more bottles of beer on the wall.')
                print()
                print('No more bottles of beer on the wall, no more bottles of beer.')
                print('Go to the store and buy some more, 99 bottles of beer on the wall.')
            elif c == '+':
                self.cnt += 1

# run code, written on HQ9+
interpreter = HQ9Plus()
interpreter.execute('H')