import pandas as pd
import numpy as np
import re
import os
import sys

from tqdm import tqdm


class IdentityAnnotator():

    def __init__(self, data_dirpath):

        self.states_path = os.path.join(data_dirpath, 'states.csv')
        self.nationalities_path = os.path.join(data_dirpath, 'nationalities.txt')
        self.ethnicities_path = os.path.join(data_dirpath, 'ethnicities.txt')
        self.terms, self.excl_terms, self.terms_re = self._pattern()

    def _pattern(self):

        # Load US states
        states = [s.lower() for s in pd.read_csv(self.states_path)['State'].tolist()]

        # Load nationalities
        with open(self.nationalities_path) as f:
            nats = [nat.lower() for nat in f.read().splitlines() if (len(nat) > 3 and not nat in states)]

        # Load ethnicities
        outlist = states + ['coast']
        with open(self.ethnicities_path) as f:
            eths = [e.split()[0].lower() for e in f.read().splitlines() if (len(e.split()[0]) > 4 and not e.split()[0].lower() in outlist)]

        # Regex patterns
        terms = {
                'age': [r'(?:[^-+\w]|^)([1-6]{1}[0-9]{1})[^-+0-9]|^([1-6]{1}[0-9]{1})$',
                       r'twelve',
                       r'thirteen',
                       r'fourteen',
                       r'fifteen',
                       r'sixteen',
                       r'seventeen',
                       r'eighteen',
                       r'nineteen',
                       r'twenty',
                       r'thirty',
                       r'forty',
                       r'fifty',
                       r'sixty'],
        #         'location': [],
                'gender': [r'male\b', r'female', 
                            r'trans', r'ftm', r'mtf', r'cis',
                            r'girl\b', r'boy\b', r'\bman\b', r'guy\b', r'woman', r'gu+rl', r'gii+rl',
                            r'non-binary', r'nonbinary', r'nb', r'agender', r'neutrois',
                            r'\bmom\b', r'\bdad\b', r'wife', r'husband', r'\bbrother\b', r'\bson\b', r'\bsister\b',
                            r'bigender', r'lgbt'],
                'sexual orientation': 
                             [r'gay', r'straight', r'lesbian', r'\bhomo',
                               r'bisexual', r'\bbi\b', r'pansexual', r'\bpan\b',
                               r'lgbt', r'queer',
                               r'\bace\b', r'\basexual', r'aro-ace', r'aro/ace',
                             ],
                 'pronouns': [
                     r'(?:\W|\b)she(?:\W|\b)', r'(?:\W|\b)her(?:\W|\b)',
                     r'(?:\W|\b)he(?:\W|\b)', r'(?:\W|\b)him(?:\W|\b)',
                     r'(?:\W|\b)they(?:\W|\b)', r'(?:\W|\b)them(?:\W|\b)',
                     r'pronouns'
                        ],
                'personality type': [
                    r'(?:i|e|a)(?:s|n)(?:t|f)(?:j|p)',
                    r'introvert',
                    r'extrovert', 
                    r'ambivert',
                    r'\b[0-9]w[0-9]\b',
                    ],
                'ethnicity/nationality': [r'\b{}\b'.format(el) for el in eths + nats] + 
                        [r'latino', r'latina', r'cubana', r'cubano', r'chilena', r'chileno', r'mexicano', r'mexicana',
                        r'palestinian'],
                'relationship status': [
                    r'taken', r'married', r'single', r'engaged', r'husband', r'spouse', r'wife', r'newlywed',
                    r'in a rl', r'in rl', r'in a relationship',
                ]
        }
        terms['sexuality/gender'] = terms['gender'] + terms['sexual orientation'] + terms['pronouns']

        excl_terms = {
            'age': ['nsfw 18', '18 nsfw', '18 only', 'only 18', '18+'],
        }

        # Combine terms in regex
        terms_re = {}
        for cat in terms:
            terms_re[cat] = r'|'.join(terms[cat])

        return terms, excl_terms, terms_re

    def _has_category(self, cat, desc, is_list_desc):
        ans = False
        
        if is_list_desc and not isinstance(desc, list):
            return ans

        elif not is_list_desc and not isinstance(desc, str):
            return ans
        
        if is_list_desc:
            ans = any(re.search(self.terms_re[cat], s) for s in desc)
                
            if cat in self.excl_terms:
                for c in self.excl_terms[cat]:
                    if any(c in s for s in desc):
                        ans = False

        else:
            ans = re.search(self.terms_re[cat], desc) is not None

            if cat in self.excl_terms:
                for c in self.excl_terms[cat]:
                    if c in desc:
                        ans = False
                
        return ans

    def annotate(self, descs, desc_colname, list_desc):
        # Annotate for identity categories
        for cat in self.terms:
            print(cat)
            descs[cat] = list(map(lambda desc: self._has_category(cat, desc, list_desc), tqdm(descs[desc_colname])))

        return descs


def main():

    # I/O files
    data_dirpath = '/usr2/mamille2/tumblr/data'
    descs_path = os.path.join(data_dirpath, 'blog_descriptions_recent100_100posts.pkl')
    outpath = os.path.join(data_dirpath, 'blog_descriptions_recent100_100posts.pkl')

    # Settings
    desc_colname = 'parsed_blog_description'
    list_desc = False

    print("Loading blog descriptions...", end=' ')
    sys.stdout.flush()
    descs = pd.read_pickle(descs_path)
    print('done.')
    sys.stdout.flush()

    print("Annotating identity categories...")
    sys.stdout.flush()
    ia = IdentityAnnotator(data_dirpath)
    descs_annotated = ia.annotate(descs, desc_colname, list_desc)
    print('done.')
    sys.stdout.flush()

    descs_annotated.to_pickle(outpath)
    print("Saved annotated data to {}".format(outpath))
    sys.stdout.flush()

if __name__=='__main__': main()
