import csv
import sys
sys.path.append('/Users/matthewdixon/Downloads/dq-MM/')
import numpy as np
from tgym.gens import CSVStreamer


def test_csv_streamer():
   # with open('../../data/AMZN-L1.csv', 'w+') as csvfile:
   #     csv_test = csv.writer(csvfile)
   #     for i in range(10):
   #         csv_test.writerow([1] * 10)
    csvstreamer = CSVStreamer(filename='../../data/AMZN-L1.csv')
    for i in range(10):
        print csvstreamer.next()

if __name__ == "__main__":
    test_csv_streamer()


