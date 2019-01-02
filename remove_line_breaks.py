import os
from tqdm import tqdm
import pdb

""" Remove line breaks from malformed CSVs from Pig """

data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample200/test'
#csv_fpath = os.path.join(data_dirpath, 'nonreblogs_descs_orig.tsv')
csv_fpaths = [os.path.join(data_dirpath, f) for f in sorted(os.listdir(data_dirpath))]

out_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample200/test'

ncols = 51

def main():
    
    for i, csv_fpath in enumerate(csv_fpaths):
        print(csv_fpath)

        outlines = []
        out_fpath = os.path.join(out_dirpath, csv_fpath + '_cleaned.tsv')

        # Read in line-by-line, split by '\t'
        with open(csv_fpath, 'r') as f:
            data = f.read() # would be better for memory to do line-by-line
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
            pbar = tqdm(total=len(tabbed))
            while tab_ctr < len(tabbed):
                segment = tabbed[tab_ctr]

                if segment.startswith('"'): # quoted column--in case contains \t
                    # Keep appending segments until segment ends with a "
                    quote_count = segment.count('"')
                    while quote_count % 2 != 0:
                        tab_ctr += 1
                        pbar.update(1)
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
                            pbar.update(1)
                            line = []
                            continue
                        outlines.append('\t'.join(line + [parts[0]])) # first part goes to previous line
                        line = [parts[1]] # second part goes to new line
                else:
                    line.append(segment.replace('\n', ' '))

                tab_ctr += 1
                pbar.update(1)

        # Save out outlines with line breaks
        with open(out_fpath, 'w') as f:
            f.write('\n'.join(outlines))
        tqdm.write("Wrote cleaned TSV")
        print()

if __name__ == '__main__':
    main()
