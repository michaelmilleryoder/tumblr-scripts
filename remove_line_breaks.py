import os
from tqdm import tqdm
import pdb
from multiprocessing import Pool

""" Remove line breaks from malformed CSVs from Pig """

#data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample200/nonreblogs_descs_orig'
#data_dirpath = '/usr2/mamille2/tumblr/data/sample1k/nonreblogs_descs_orig'
data_dirpath = '/usr2/mamille2/tumblr/data/sample1k/reblogs_descs_orig'
out_dirpath = '/usr2/mamille2/tumblr/data/sample1k/reblogs_descs_cleaned'
#csv_fnames = [f for f in sorted(os.listdir(data_dirpath)) if f.startswith('part')]
csv_fnames = sorted(os.listdir(data_dirpath))
#ncols = 51 # nonreblogs
ncols = 50 # reblogs

if not os.path.exists(out_dirpath):
    os.mkdir(out_dirpath)


def process_csv(csv_fname):

        pdb.set_trace()
        tqdm.write(csv_fname)
        csv_fpath = os.path.join(data_dirpath, csv_fname)

        outlines = []
        out_fpath = os.path.join(out_dirpath, csv_fname)

        # Read in line-by-line, split by '\t'
        with open(csv_fpath, 'r') as f:
            data = f.read()
            #header_idx = data.find('\n')
            #header = data[:header_idx].split('\t')
            #outlines = [header]
            #if len(header) != ncols:
            #    pdb.set_trace()
            #print(header)
            #print(ncols)

            #tabbed = data[header_idx:].split('\t') # But some segments still break lines with \n
            first_segment = data[:data.find('\t')]
            tabbed = data[data.find('\t'):].strip().split('\t') # But some segments still break lines with \n
            line = [first_segment]
            tab_ctr = 0
            #pbar = tqdm(total=len(tabbed))
            while tab_ctr < len(tabbed):
                segment = tabbed[tab_ctr]

                if segment.startswith('"'): # quoted column--in case contains \t
                    # Keep appending segments until segment ends with a "
                    quote_count = segment.count('"')
                    while quote_count % 2 != 0:
                        tab_ctr += 1
                        #pbar.update(1)
                        if tab_ctr < len(tabbed):
                            quote_count += tabbed[tab_ctr].count('"')
                            segment += ' ' + tabbed[tab_ctr]
                        else:
                            break

                if len(line) % (ncols-1) == 0: # Ending segment of a line
                    if tab_ctr == len(tabbed) - 1: # final segment of last line
                        line.append(segment.replace('\n', ' '))
                        outlines.append('\t'.join(line))
                    else:
                        parts = segment.split('\n')
                        if len(parts) != 2: # infrequent; skip line
                            tab_ctr +=1
                            #pbar.update(1)
                            line = []
                            continue
                        outlines.append('\t'.join(line + [parts[0]])) # first part goes to previous line
                        line = [parts[1]] # second part goes to new line
                else:
                    line.append(segment.replace('\n', ' '))

                tab_ctr += 1
                #pbar.update(1)

        # Save out outlines with line breaks
        with open(out_fpath, 'w') as f:
            f.write('\n'.join(outlines))
        tqdm.write("Wrote cleaned TSV")
        #print()

def main():
    
    #for csv_fname in tqdm(csv_fnames):
    #    process_csv(csv_fname)
    #map(process_csv, tqdm(csv_fnames))
    with Pool(15) as pool:
        tqdm(pool.imap(process_csv, csv_fnames), total=len(csv_fnames)) # tqdm might not work
        #pool.map(process_csv, tqdm(csv_fnames)) # tqdm might not work


if __name__ == '__main__':
    main()
