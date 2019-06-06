# # Make 1M balanced dataset
# Load data

import os
import pandas as pd
from tqdm import tqdm as tqdm

def main():
    data_dirpath = '/mnt/interns/myoder/textposts_captions/'
    communities = os.listdir(data_dirpath)
    data = {}

    for c in tqdm(communities):
        data[c] = pd.read_pickle(os.path.join(data_dirpath, c, f'{c}.pkl')).sample(int(1e5), random_state=9)

    # Add community column

    for c in communities:
        data[c]['community'] = [c] * len(data[c])

    data_1m = pd.concat(data.values())
    print(len(data_1m))
    #print(data_1m.columns)

    # Save
    print("Saving out...")
    data_1m.to_pickle('/mnt/interns/myoder/textposts_captions_1m.pkl')

if __name__ == '__main__':
    main()
