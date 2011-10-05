"""Test fspsq with a tiny grid of models--a basic burn in test case."""

from fsps import FSPSLibrary
from fsps import ParameterSet

def main():
    tinyLibrary = TinySSPGrid("tinyssp", dbname="fsps")
    tinyLibrary.reset()
    tinyLibrary.generate_grid()
    tinyLibrary.compute_models(nThreads=1)
    tinyLibrary.create_table("tiny_table.h5")

class TinySSPGrid(FSPSLibrary):
    """A small grid SSPs for three metallicities."""
    
    def generate_grid(self):
        """Create the model grid."""
        zmets = [1,2,3]
        for i, zmet in enumerate(zmets):
            pset = ParameterSet(None, sfh=0, zmet=zmet)
            self.register_pset(pset)
            
if __name__ == '__main__':
    main()
