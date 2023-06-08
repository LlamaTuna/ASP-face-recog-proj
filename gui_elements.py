from PyQt5.QtWidgets import QTableWidgetItem

class NumericTableWidgetItem(QTableWidgetItem):
    def __init__(self, value):
        super().__init__(str(value))

    def __lt__(self, other):
        my_number = float(self.text().rstrip('%'))
        other_number = float(other.text().rstrip('%'))
        return my_number < other_number
  
class MatchTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        my_number = int(self.text().split()[-1])
        other_number = int(other.text().split()[-1])
        return my_number < other_number