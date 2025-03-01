class Cell:
    def __init__(self, index):
    
        self.index: int = index
        self.name: str = ""
        self.attractor_dict: dict = {}
        self.expression: dict = {}
        self.simulation_results: list= []
        self.attractor_analysis_cluster: int = -1
        self.groups: str = ""

class CellPopulation:
    def __init__(self, cells):
        self.cells = cells