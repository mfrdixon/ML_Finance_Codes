import csv

import numpy as np
from tgym.core import DataGenerator


class CSVStreamer(DataGenerator):
    """Data generator from csv file.
    The csv file should no index columns.

    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    @staticmethod
    def _generator(filename, header=False):
        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile)
            if header:
                next(reader, None)
            for row in reader:
                #assert len(row) % 2 == 0
                yield np.array(row, dtype=np.float)

    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        print("End of data reached, rewinding.")
        super(self.__class__, self).rewind()

#    def rewind(self):
#        """For this generator, we want to rewind only when the end of the data is reached.
#        """
#        pass
