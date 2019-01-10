import copy
import csv
import re
import sys

def paren_process(s):
    lparen_count = 0
    rparen_count = 0
    for i in range(1, len(s)):
        if s[i] == '(' and s[i-1] != ':' and s[i-1] != ';' and s[i-1] != '-' and s[i-1] != '(':
            lparen_count += 1
        if s[i] == ')' and s[i-1] != ':' and s[i-1] != ';' and s[i-1] != '-' and s[i-1] != ')':
            rparen_count += 1

    if rparen_count == lparen_count:
         return s[1:]

    return s[1:-1]

def main():
    reader = csv.reader(open(sys.argv[1], 'r'))
    reprocessed = []
    labels = next(reader)
    table = str.maketrans('{}','[]')

    paren_in_tag_re = '[,{]\(.*?\)[,}]'

    for row in reader:
        if row[3] == 'nan':
            row[3] = '{}'
        matches = re.findall(paren_in_tag_re, row[3])
        matches = [paren_process(x[1:-1]) for x in matches]
        row[3] = matches

        if row[5] == 'nan':
            row[5] = 0.0

        if row[6] == 'nan':
            row[6] = ''

        if row[7] == 'nan':
            row[7] = ''

        reprocessed.append(row)

    writer = csv.writer(open(sys.argv[1] + '_reprocessed', 'w'))
    writer.writerow(labels)
    writer.writerows(reprocessed)

if __name__ == '__main__': main()
